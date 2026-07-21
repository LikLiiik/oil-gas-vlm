"""VLM API 后端单元测试。

所有 API 调用都用 mock，不产生真实费用，也不要求网络/GPU。
本测试文件同时承担"API 模式不加载 torch"的回归保护。

    python test/test_vlm_api_unit.py
"""
from __future__ import annotations

import base64
import importlib
import inspect
import io
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from pipeline import Pipeline, downstream
from pipeline.agents import AgentResult
from pipeline.downstream.base import _REGISTRY
from pipeline.vlm import VLMClient, VLMResponse, extract_json


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _make_pil(seed: int = 0, w: int = 64, h: int = 64, mode: str = "RGB"):
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode=mode).convert(mode)


def _img(seed: int = 0):
    return _make_pil(seed=seed)


def _img_size(w: int, h: int, seed: int = 0):
    return _make_pil(seed=seed, w=w, h=h)


def _ok_response(text: str, model: str = "qwen3-vl-plus"):
    """构造一个看起来像 OpenAI ChatCompletion 的对象。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        model=model,
    )


def _empty_response(model: str = "qwen3-vl-plus"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=""))],
        model=model,
    )


def _error_response(status: int, msg: str = "mock error",
                    err_type: str = "APIStatusError"):
    """构造一个 openai SDK 风格异常（带 status_code）。"""
    err = type(err_type, (Exception,), {})("{}: {}".format(status, msg))
    err.status_code = status
    return err


# ---------------------------------------------------------------------------
# 1) PIL → base64 data URL
# ---------------------------------------------------------------------------

def test_pil_to_data_url_png():
    from pipeline.vlm_backends.openai_compatible import _encode_pil_to_data_url
    img = _img()
    url, size, n_bytes = _encode_pil_to_data_url(img, image_format="PNG")
    assert size == img.size
    assert n_bytes > 0
    assert url.startswith("data:image/png;base64,")
    payload = url.split(",", 1)[1]
    decoded = base64.b64decode(payload)
    # 重新打开应得到等价图
    re = Image.open(io.BytesIO(decoded))
    assert re.size == img.size
    assert re.format == "PNG"


def test_pil_to_data_url_jpeg():
    from pipeline.vlm_backends.openai_compatible import _encode_pil_to_data_url
    img = _img()
    url, size, n_bytes = _encode_pil_to_data_url(
        img, image_format="JPEG", jpeg_quality=80,
    )
    assert url.startswith("data:image/jpeg;base64,")
    decoded = base64.b64decode(url.split(",", 1)[1])
    re = Image.open(io.BytesIO(decoded))
    assert re.size == img.size
    assert re.format == "JPEG"


def test_pil_resize_keeps_aspect():
    from pipeline.vlm_backends.openai_compatible import _encode_pil_to_data_url
    img = _img_size(w=200, h=100)
    # max_edge=50 → 50×25
    url, size, _ = _encode_pil_to_data_url(
        img, image_format="PNG", max_edge=50,
    )
    assert size == (50, 25)
    # 0 / None → 不缩放
    _, size2, _ = _encode_pil_to_data_url(img, image_format="PNG", max_edge=0)
    assert size2 == (200, 100)


def test_pil_to_data_url_rejects_bad_format():
    from pipeline.vlm_backends.openai_compatible import _encode_pil_to_data_url
    try:
        _encode_pil_to_data_url(_img(), image_format="TIFF")
    except ValueError as e:
        assert "PNG" in str(e) and "JPEG" in str(e)
    else:
        raise AssertionError("expected ValueError for bad image_format")


# ---------------------------------------------------------------------------
# 2) 多图顺序保持
# ---------------------------------------------------------------------------

def test_multi_image_order_preserved():
    """传入 [a, b, c] 时，发送给 API 的 image_url 列表也应是 [a, b, c]。"""
    from pipeline.vlm_backends.openai_compatible import OpenAICompatibleVLMBackend

    images = [_make_pil(seed=1), _make_pil(seed=2), _make_pil(seed=3)]
    captured = {}

    def fake_create(**kwargs):
        # 取出 messages[1].content 中的 image_url 顺序
        msgs = kwargs["messages"]
        user = next(m for m in msgs if m["role"] == "user")
        captured["image_urls"] = [
            c["image_url"]["url"] for c in user["content"]
            if c.get("type") == "image_url"
        ]
        captured["text"] = next(
            c["text"] for c in user["content"] if c.get("type") == "text"
        )
        return _ok_response("ok")

    with patch.object(OpenAICompatibleVLMBackend, "_build_client") as bc:
        bc.return_value = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create),
        ))
        backend = OpenAICompatibleVLMBackend(
            api_key="x", base_url="https://x/v1",
            model="m", max_retries=0, max_image_edge=0,
        )
        backend.call("sys", images, "hello")

    assert len(captured["image_urls"]) == 3
    # 每张图的 base64 payload 互不相同
    payloads = [u.split(",", 1)[1] for u in captured["image_urls"]]
    assert len(set(payloads)) == 3
    # 文本在最后
    assert captured["text"] == "hello"


# ---------------------------------------------------------------------------
# 3) API 后端不加载 torch
# ---------------------------------------------------------------------------

def test_api_backend_source_does_not_import_torch():
    """openai_compatible.py 的源码 AST 里不应该 import torch / transformers。

    注意：模块 docstring 里可以自由写"import torch"这种字面英文
    （作为反例说明），所以这里用 AST 解析而不是字符串匹配。
    """
    import ast
    from pipeline.vlm_backends import openai_compatible as oc
    tree = ast.parse(inspect.getsource(oc))

    forbidden_modules = {"torch", "transformers"}
    forbidden_names = {"Qwen3VLForConditionalGeneration",
                       "AutoProcessor"}

    def _check(node):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top not in forbidden_modules, (
                    f"{oc.__name__} imports {alias.name!r} "
                    f"(forbidden in API backend)"
                )
                assert alias.name not in forbidden_names
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            assert mod not in forbidden_modules, (
                f"{oc.__name__} does 'from {node.module} import ...' "
                f"(forbidden in API backend)"
            )
            for alias in node.names:
                assert alias.name not in forbidden_names

    for node in ast.walk(tree):
        _check(node)


def test_api_backend_module_import_does_not_trigger_torch():
    """即使 torch 之前已被加载（比如 test_pipeline_unit 之前跑过），
    import openai_compatible 这件事本身**不会**主动加载 torch / transformers。

    这个测试通过 importlib 强制重导来检查 sys.modules 没有新增 torch 引用。
    """
    mod_name = "pipeline.vlm_backends.openai_compatible"
    saved = sys.modules.get(mod_name)
    if saved is not None:
        del sys.modules[mod_name]
    try:
        importlib.import_module(mod_name)
        # 不强制要求 torch 不在 sys.modules（它可能已被别的测试加载）；
        # 但要求这个 module 的导入路径里**没有** torch。
        mod = sys.modules[mod_name]
        # 模块的 globals / __dict__ 不应有 torch
        assert "torch" not in mod.__dict__
    finally:
        if saved is not None:
            sys.modules[mod_name] = saved


# ---------------------------------------------------------------------------
# 4) 环境变量后端切换
# ---------------------------------------------------------------------------

def test_env_var_selects_backend(monkeypatch=None):
    """未指定 backend 参数时，按 VLM_BACKEND env 选。"""
    monkeypatch = monkeypatch or _MonkeyPatch()
    try:
        # default = local
        monkeypatch.setenv("VLM_BACKEND", None)
        from pipeline.vlm import VLMClient
        v = VLMClient()
        assert v.backend_name == "local"
        assert type(v.backend).__name__ == "LocalQwenVLMBackend"

        # env=api -> api backend
        monkeypatch.setenv("VLM_API_KEY", "k1")
        monkeypatch.setenv("VLM_BACKEND", "api")
        v2 = VLMClient()
        assert v2.backend_name == "api"
        assert type(v2.backend).__name__ == "OpenAICompatibleVLMBackend"
    finally:
        monkeypatch.restore()


def test_invalid_backend_raises():
    from pipeline.vlm import VLMClient
    try:
        VLMClient(backend="bogus")
    except ValueError as e:
        msg = str(e)
        assert "bogus" in msg
        assert "local" in msg and "api" in msg
    else:
        raise AssertionError("expected ValueError for bogus backend")


# ---------------------------------------------------------------------------
# 5) API Key 缺失时报清晰错误
# ---------------------------------------------------------------------------

def test_api_key_missing_raises(monkeypatch=None):
    monkeypatch = monkeypatch or _MonkeyPatch()
    try:
        monkeypatch.delenv("VLM_API_KEY")
        monkeypatch.delenv("DASHSCOPE_API_KEY")
        from pipeline.vlm_backends.openai_compatible import OpenAICompatibleVLMBackend
        try:
            OpenAICompatibleVLMBackend()
        except RuntimeError as e:
            msg = str(e)
            # 错误信息应提示用户设哪个 env
            assert "VLM_API_KEY" in msg or "DASHSCOPE_API_KEY" in msg
        else:
            raise AssertionError("expected RuntimeError for missing API key")
    finally:
        monkeypatch.restore()


def test_api_key_priority_dashscope_first():
    """未设 VLM_API_KEY 时，回退到 DASHSCOPE_API_KEY。"""
    from pipeline.vlm_backends.openai_compatible import (
        OpenAICompatibleVLMBackend, _resolve_api_key,
    )
    mp = _MonkeyPatch()
    try:
        mp.delenv("VLM_API_KEY")
        mp.setenv("DASHSCOPE_API_KEY", "ds-key-1234")
        assert _resolve_api_key() == "ds-key-1234"
        # 但 VLM_API_KEY 优先
        mp.setenv("VLM_API_KEY", "main-key-9999")
        assert _resolve_api_key() == "main-key-9999"
        b = OpenAICompatibleVLMBackend()
        assert b.api_key == "main-key-9999"
    finally:
        mp.restore()


# ---------------------------------------------------------------------------
# 6) mock OpenAI Client 模拟正常响应
# ---------------------------------------------------------------------------

def test_api_normal_text_response():
    from pipeline.vlm_backends.openai_compatible import OpenAICompatibleVLMBackend
    backend = OpenAICompatibleVLMBackend(
        api_key="k", base_url="https://x/v1", model="m", max_retries=0,
    )
    with patch.object(backend, "_build_client") as bc:
        bc.return_value = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **k: _ok_response("hello")),
        ))
        text, elapsed = backend.call("sys", [], "hi")
    assert text == "hello"
    assert elapsed >= 0.0


def test_api_passes_max_tokens_and_temperature():
    from pipeline.vlm_backends.openai_compatible import OpenAICompatibleVLMBackend
    backend = OpenAICompatibleVLMBackend(
        api_key="k", base_url="https://x/v1", model="m", max_retries=0,
    )
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _ok_response("ok")

    with patch.object(backend, "_build_client") as bc:
        bc.return_value = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create),
        ))
        backend.call("sys", [], "hi", max_new_tokens=123, temperature=0.5)
    assert captured["max_tokens"] == 123
    assert captured["temperature"] == 0.5
    assert captured["model"] == "m"


def test_api_json_mode_off_by_default():
    """默认不开 response_format，避免锁定到特定服务。"""
    from pipeline.vlm_backends.openai_compatible import OpenAICompatibleVLMBackend
    backend = OpenAICompatibleVLMBackend(
        api_key="k", base_url="https://x/v1", max_retries=0,
    )
    captured = {}
    with patch.object(backend, "_build_client") as bc:
        bc.return_value = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **k: (
                captured.update(k) or _ok_response("ok")
            )),
        ))
        backend.call("sys", [], "hi")
    assert "response_format" not in captured


def test_api_json_mode_on_when_enabled():
    from pipeline.vlm_backends.openai_compatible import OpenAICompatibleVLMBackend
    backend = OpenAICompatibleVLMBackend(
        api_key="k", base_url="https://x/v1",
        max_retries=0, json_mode=True,
    )
    captured = {}
    with patch.object(backend, "_build_client") as bc:
        bc.return_value = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **k: (
                captured.update(k) or _ok_response("ok")
            )),
        ))
        backend.call("sys", [], "hi")
    assert captured.get("response_format") == {"type": "json_object"}


# ---------------------------------------------------------------------------
# 7) API 返回空内容 → 抛错（让 schema retry 接管）
# ---------------------------------------------------------------------------

def test_api_empty_content_raises():
    from pipeline.vlm_backends.openai_compatible import OpenAICompatibleVLMBackend
    backend = OpenAICompatibleVLMBackend(
        api_key="k", base_url="https://x/v1", max_retries=0,
    )
    with patch.object(backend, "_build_client") as bc:
        bc.return_value = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **k: _empty_response()),
        ))
        try:
            backend.call("sys", [], "hi")
        except RuntimeError as e:
            msg = str(e)
            assert "empty" in msg.lower()
            # 错误信息应该含 model / host / images
            assert "m" in msg or "qwen" in msg
        else:
            raise AssertionError("expected RuntimeError on empty content")


# ---------------------------------------------------------------------------
# 8) API 网络错误 + 有限重试
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    def __init__(self, msg="timeout"):
        super().__init__(msg)


def test_api_5xx_retries_then_raises():
    """5xx 是可重试；max_retries=2 表示最多 3 次总尝试。"""
    from pipeline.vlm_backends.openai_compatible import OpenAICompatibleVLMBackend
    backend = OpenAICompatibleVLMBackend(
        api_key="k", base_url="https://x/v1", max_retries=2,
    )
    n_calls = {"n": 0}
    sleep_calls = []

    def fake_create(**k):
        n_calls["n"] += 1
        raise _error_response(503, "upstream busy")

    with patch.object(backend, "_build_client") as bc, \
         patch("pipeline.vlm_backends.openai_compatible.time.sleep",
               lambda s: sleep_calls.append(s)):
        bc.return_value = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create),
        ))
        try:
            backend.call("sys", [], "hi")
        except RuntimeError as e:
            assert "503" in str(e)
        else:
            raise AssertionError("expected RuntimeError after retries")

    # 初次 + 2 重试 = 3 次
    assert n_calls["n"] == 3
    # 退避 1s, 2s
    assert sleep_calls == [1.0, 2.0]


def test_api_429_retries_with_backoff():
    from pipeline.vlm_backends.openai_compatible import OpenAICompatibleVLMBackend
    backend = OpenAICompatibleVLMBackend(
        api_key="k", base_url="https://x/v1", max_retries=2,
    )
    n_calls = {"n": 0}
    sleeps = []

    def fake_create(**k):
        n_calls["n"] += 1
        if n_calls["n"] < 3:
            raise _error_response(429, "rate limited")
        return _ok_response("recovered")

    with patch.object(backend, "_build_client") as bc, \
         patch("pipeline.vlm_backends.openai_compatible.time.sleep",
               lambda s: sleeps.append(s)):
        bc.return_value = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create),
        ))
        text, _ = backend.call("sys", [], "hi")
    assert text == "recovered"
    assert n_calls["n"] == 3
    assert sleeps == [1.0, 2.0]


def test_api_401_does_not_retry():
    """401 是配置错误，重试没意义。"""
    from pipeline.vlm_backends.openai_compatible import OpenAICompatibleVLMBackend
    backend = OpenAICompatibleVLMBackend(
        api_key="k", base_url="https://x/v1", max_retries=5,
    )
    n_calls = {"n": 0}

    def fake_create(**k):
        n_calls["n"] += 1
        raise _error_response(401, "invalid api key")

    with patch.object(backend, "_build_client") as bc, \
         patch("pipeline.vlm_backends.openai_compatible.time.sleep"):
        bc.return_value = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create),
        ))
        try:
            backend.call("sys", [], "hi")
        except RuntimeError as e:
            assert "401" in str(e)
        else:
            raise AssertionError("expected RuntimeError for 401")
    assert n_calls["n"] == 1   # 立即放弃，不重试


def test_api_timeout_is_retryable():
    """APITimeoutError 名字含 timeout，应被识别为可重试。"""
    from pipeline.vlm_backends.openai_compatible import OpenAICompatibleVLMBackend
    backend = OpenAICompatibleVLMBackend(
        api_key="k", base_url="https://x/v1", max_retries=1,
    )
    n_calls = {"n": 0}
    sleeps = []

    class APITimeoutError(Exception):
        pass

    def fake_create(**k):
        n_calls["n"] += 1
        if n_calls["n"] == 1:
            raise APITimeoutError("timed out")
        return _ok_response("ok")

    with patch.object(backend, "_build_client") as bc, \
         patch("pipeline.vlm_backends.openai_compatible.time.sleep",
               lambda s: sleeps.append(s)):
        bc.return_value = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create),
        ))
        text, _ = backend.call("sys", [], "hi")
    assert text == "ok"
    assert n_calls["n"] == 2
    assert sleeps == [1.0]


# ---------------------------------------------------------------------------
# 9) API 返回合法 JSON
# ---------------------------------------------------------------------------

def test_call_json_parses_valid_json():
    """call_json 应该把响应里的 JSON 抽出来填到 data。"""
    from pipeline.vlm_backends.openai_compatible import OpenAICompatibleVLMBackend
    from pipeline.vlm import _call_json_with_backend
    backend = OpenAICompatibleVLMBackend(
        api_key="k", base_url="https://x/v1", max_retries=0,
    )
    payload = {"scene_understanding": "test", "workflow_steps": []}
    raw = "noise\n" + __import__("json").dumps(payload, ensure_ascii=False) + "\nmore"
    with patch.object(backend, "_build_client") as bc:
        bc.return_value = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **k: _ok_response(raw)),
        ))
        resp = _call_json_with_backend(
            backend, "sys", [], "hi", schema=None, max_new_tokens=512,
        )
    assert resp.attempts == 1
    assert resp.schema_valid is True
    assert resp.data == payload
    assert resp.text == raw


# ---------------------------------------------------------------------------
# 10) API 返回 Markdown 包裹 JSON
# ---------------------------------------------------------------------------

def test_call_json_parses_markdown_wrapped_json():
    from pipeline.vlm_backends.openai_compatible import OpenAICompatibleVLMBackend
    from pipeline.vlm import _call_json_with_backend
    backend = OpenAICompatibleVLMBackend(
        api_key="k", base_url="https://x/v1", max_retries=0,
    )
    raw = "Here is the answer:\n```json\n{\"a\": 1, \"b\": [1, 2]}\n```\nDone."
    with patch.object(backend, "_build_client") as bc:
        bc.return_value = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **k: _ok_response(raw)),
        ))
        resp = _call_json_with_backend(
            backend, "sys", [], "hi", schema=None,
        )
    assert resp.schema_valid is True
    assert resp.data == {"a": 1, "b": [1, 2]}


# ---------------------------------------------------------------------------
# 11) Schema 失败 → 一次修复重试
# ---------------------------------------------------------------------------

def test_schema_failure_triggers_one_repair_retry():
    """第一次响应 schema 不过 -> 重试一次（带上一份 JSON + 错误）。"""
    from pipeline.vlm_backends.openai_compatible import OpenAICompatibleVLMBackend
    from pipeline.vlm import _call_json_with_backend
    backend = OpenAICompatibleVLMBackend(
        api_key="k", base_url="https://x/v1", max_retries=0,
    )
    # 用一个明确 schema：要求 required 字段 "answer"
    schema = {
        "type": "object",
        "required": ["answer"],
        "properties": {"answer": {"type": "string"}},
    }
    bad = {"wrong_field": 1}
    good = {"answer": "ok"}
    responses = [
        _ok_response("noise " + __import__("json").dumps(bad)),
        _ok_response("again " + __import__("json").dumps(good)),
    ]
    n = {"i": 0}
    with patch.object(backend, "_build_client") as bc:
        bc.return_value = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **k: responses[n["i"]] or
                                        (n.update(i=n["i"]+1) or responses[0])),
        ))
        # 简单版：每次返回 responses[0] 然后切到 responses[1]
        calls = {"v": 0}
        def serve(**k):
            i = calls["v"]; calls["v"] += 1
            return responses[i]
        with patch.object(backend, "_build_client") as bc2:
            bc2.return_value = SimpleNamespace(chat=SimpleNamespace(
                completions=SimpleNamespace(create=serve),
            ))
            resp = _call_json_with_backend(
                backend, "sys", [], "hi", schema=schema,
            )
    assert resp.attempts == 2
    assert resp.schema_valid is True
    assert resp.data == good
    assert calls["v"] == 2


# ---------------------------------------------------------------------------
# 12) 网络重试 + Schema 重试不会无限循环
# ---------------------------------------------------------------------------

def test_no_infinite_loop_with_both_retries():
    """即使 network 重试 + schema 重试都失败，总次数也应有界。"""
    from pipeline.vlm_backends.openai_compatible import OpenAICompatibleVLMBackend
    from pipeline.vlm import _call_json_with_backend
    backend = OpenAICompatibleVLMBackend(
        api_key="k", base_url="https://x/v1",
        max_retries=2,  # 初次+2=3 次
    )
    schema = {"type": "object", "required": ["answer"],
              "properties": {"answer": {"type": "string"}}}
    bad = {"wrong_field": 1}
    raw_bad = "noise " + __import__("json").dumps(bad)

    n_calls = {"n": 0}

    def fake_create(**k):
        n_calls["n"] += 1
        # 第一次：503（重试 2 次后再成功）
        if n_calls["n"] in (1, 2):
            raise _error_response(503, "upstream busy")
        # 第 3 次：成功但 schema 失败 → call_json 会再调用 1 次
        if n_calls["n"] == 3:
            return _ok_response(raw_bad)
        # 第 4 次：再失败一次 schema，但 call_json 不再重试（attempts 已经 = 2）
        return _ok_response(raw_bad)

    with patch.object(backend, "_build_client") as bc, \
         patch("pipeline.vlm_backends.openai_compatible.time.sleep"):
        bc.return_value = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create),
        ))
        resp = _call_json_with_backend(
            backend, "sys", [], "hi", schema=schema,
        )
    # 网络层 3 次 + schema retry 1 次 = 总 4 次
    assert n_calls["n"] == 4
    # 仍然给出 data（schema 不通过时也保留）
    assert resp.attempts == 2   # schema 层只看自己
    assert resp.data == bad
    assert resp.schema_valid is False


# ---------------------------------------------------------------------------
# 13) 本地后端已有行为不被破坏
# ---------------------------------------------------------------------------

def test_local_backend_still_works():
    """LocalQwenVLMBackend 的接口和原 VLMClient 一致：load/call + 错误。"""
    from pipeline.vlm_backends.local_qwen import LocalQwenVLMBackend
    b = LocalQwenVLMBackend()  # 不传 path
    # 没配 QWEN_VL_PATH 又没显式 path -> 首次 load 必报清晰错误
    try:
        b.load()
    except RuntimeError as e:
        assert "QWEN_VL_PATH" in str(e)
    else:
        raise AssertionError("expected RuntimeError for missing model path")


# ---------------------------------------------------------------------------
# 14) Pipeline(vlm=fake_vlm) 仍然兼容
# ---------------------------------------------------------------------------

def test_pipeline_accepts_fake_vlm():
    """老用法：Pipeline(vlm=fake_vlm) 仍应工作（duck typing 兼容）。"""
    class FakeVLM:
        def __init__(self, data):
            self._data = data
        def call_json(self, *a, **kw):
            return VLMResponse(
                text=__import__("json").dumps(self._data),
                data=self._data, elapsed_s=0.0, attempts=1,
                schema_valid=True, schema_errors=[],
            )
    vlm = FakeVLM({"scene_understanding": "x", "workflow_steps": []})
    p = Pipeline(vlm=vlm, verbose=False)
    assert p.vlm is vlm


# ---------------------------------------------------------------------------
# 15) Mock 端到端 Pipeline 测试：API 后端 + Pipeline.run_from_adapter
# ---------------------------------------------------------------------------

def _make_minimal_run_dir(tmp: Path) -> Path:
    """构造一个最小可被 Pipeline.run_from_adapter 接受的 run 目录。

    不依赖真实 SEG-Y 文件——只生成符合 manifest/schema 契约的最小集，
    并塞入一张合成 PNG 让 image 能被 PIL 打开。
    """
    import json as _json
    from PIL import Image as _Image
    run = tmp / "fake_sample"
    (run / "assets" / "seismic").mkdir(parents=True)
    (run / "assets" / "well_logs").mkdir(parents=True)
    (run / "prompts").mkdir()
    (run / "schemas").mkdir()

    img_path = run / "assets" / "seismic" / "inline_model.png"
    _Image.new("RGB", (32, 32), (0, 0, 0)).save(img_path)

    (run / "prompts" / "system_prompt.txt").write_text(
        "你是地球物理 AI 助手。", encoding="utf-8",
    )
    (run / "prompts" / "user_prompt.txt").write_text(
        "分析这张图。仅输出 JSON。", encoding="utf-8",
    )
    (run / "manifest.json").write_text(_json.dumps({
        "schema_version": "1.0",
        "sample_id": "fake_sample",
        "task": {
            "type": "geological_target_detection",
            "target_classes": ["fault"],
        },
        "run_mode": "fake",
        "availability": {"seismic": True, "well_logs": False},
        "alignment": {"fusion_permission": "seismic_only"},
        "seismic": {
            "shape": [4, 4, 4],
            "domain": "time",
            "crs": {"name": "fake"},
            "views": {
                "inline": {
                    "physical_view": "inline",
                    "array_shape": [4, 4],
                    "axis_labels": ["crossline_index", "sample_index"],
                    "source_indices": {"inline_index": 1},
                    "model_image_path": "assets/seismic/inline_model.png",
                },
            },
        },
        "well_logs": {"available": False, "curves_present": []},
    }, ensure_ascii=False), encoding="utf-8")
    (run / "request.json").write_text(_json.dumps({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "name": "seismic_inline",
                 "path": "assets/seismic/inline_model.png",
                 "physical_view": "inline"},
            ]},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    # schema 至少要有 downstream_plan 字段
    (run / "schemas" / "expected_model_output.schema.json").write_text(_json.dumps({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["downstream_plan"],
        "properties": {
            "downstream_plan": {
                "type": "object",
                "required": ["scene_understanding", "workflow_steps"],
                "properties": {
                    "scene_understanding": {"type": "string"},
                    "max_iterations": {"type": "integer"},
                    "workflow_steps": {"type": "array"},
                },
            },
        },
    }, ensure_ascii=False), encoding="utf-8")
    return run


def test_end_to_end_pipeline_with_api_backend_mock():
    """完整 Pipeline.run_from_adapter + API 后端 + mock OpenAI client。

    验证：
    1. API 后端能驱动完整的 plan -> execute -> verify -> 聚合输出流程
    2. 多个 VLM 调用（plan 1 + verify N）通过同一个 client 顺序发出
    3. 失败时不破坏 Pipeline 状态
    """
    import json as _json
    import tempfile
    from PIL import Image as _Image

    from pipeline.vlm_backends.openai_compatible import OpenAICompatibleVLMBackend

    # 1) 准备 run 目录
    with tempfile.TemporaryDirectory() as td:
        run_dir = _make_minimal_run_dir(Path(td))

        # 2) 临时注册一个名为 "sam" 的 stub（覆盖原 sam）—— 用 enum 里允许的
        #    名以便通过 WORKFLOW_PLAN_SCHEMA 校验；测试结束会恢复。
        from pipeline.downstream.base import _REGISTRY
        original_sam = _REGISTRY.get("sam")

        class _Stub:
            name = "sam"
            description = "stub for end-to-end test"
            required_fields = []
            output_shape = ""
            def detect(self, instruction, image=None, context=None):
                return [{
                    "id": "ab_stub_s1_i0", "det_id": "ab_stub_s1_i0",
                    "class_name": "fault", "bbox_pixel": [4, 4, 12, 12],
                    "bbox_norm": [0.125, 0.125, 0.375, 0.375],
                    "confidence": 0.9, "model": "sam",
                }]
        from pipeline import downstream
        downstream.register(_Stub())

        # 3) 准备 mock OpenAI 响应
        #    满足 WORKFLOW_PLAN_SCHEMA：scene_understanding + workflow_steps (model=sam, 合法 instruction)
        plan_data = {
            "scene_understanding": "fake inline",
            "max_iterations": 1,
            "workflow_steps": [{
                "step": 1, "model": "sam",
                "image_name": "seismic_inline",
                "instruction": {
                    "prompt_type": "bbox",
                    "prompt_value": [4, 4, 12, 12],
                    "label": "fault",
                },
                "reason": "unit test",
            }],
        }
        ver_data = {
            "verified": [
                {"result_id": "ab_stub_s1_i0", "is_real": True,
                 "confidence": 0.95, "rejection_reason": ""},
            ],
            "need_retry": False,
        }
        plan_raw = "noise\n" + _json.dumps(plan_data) + "\nmore"
        ver_raw = "again\n" + _json.dumps(ver_data) + "\nfinal"

        responses = [_ok_response(plan_raw), _ok_response(ver_raw)]
        n = {"i": 0}
        call_log: list[str] = []

        def fake_create(**kwargs):
            call_log.append("call")
            i = n["i"]; n["i"] += 1
            return responses[i]

        # 4) 构造 VLMClient(backend='api') + Pipeline
        # （测试场景下需要临时给个假 key，因为 OpenAICompatibleVLMBackend
        # 构造时就要求 key；key 不会真用，因为我们 mock 了 client）
        mp = _MonkeyPatch()
        mp.setenv("VLM_API_KEY", "test-fake-key")
        try:
            vlm = VLMClient(backend="api")
            # 把 backend 的 _build_client 替换掉
            with patch.object(vlm.backend, "_build_client") as bc:
                bc.return_value = SimpleNamespace(chat=SimpleNamespace(
                    completions=SimpleNamespace(create=fake_create),
                ))
                p = Pipeline(vlm=vlm, verbose=False)
                out_dir = Path(td) / "out"
                report = p.run_from_adapter(
                    run_dir=str(run_dir),
                    out_dir=str(out_dir),
                    verify=True, max_iterations=1,
                )

            # 5) 断言
            # 调用了 2 次：1 次规划 + 1 次验证
            assert len(call_log) == 2, f"expected 2 VLM calls, got {len(call_log)}"
            assert report.get("ok") is True, f"report not ok: {report}"
            plan = report.get("vlm_plan") or {}
            assert plan.get("scene_understanding") == "fake inline"
            # 下游确实执行了
            downstream_section = report.get("downstream") or {}
            assert downstream_section.get("n_detections", 0) >= 1
        finally:
            # 清理
            _REGISTRY.pop("sam", None)
            if original_sam is not None:
                _REGISTRY["sam"] = original_sam
            mp.restore()


# ---------------------------------------------------------------------------
# 辅助：极简 monkeypatch
# ---------------------------------------------------------------------------

class _MonkeyPatch:
    """比 pytest.MonkeyPatch 还轻的版本，本文件不依赖 pytest。"""
    def __init__(self):
        self._saved = {}

    def setenv(self, name, value):
        if name not in self._saved:
            self._saved[name] = ("SET", os.environ.get(name))
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def delenv(self, name):
        if name not in self._saved:
            self._saved[name] = ("SET", os.environ.get(name))
        os.environ.pop(name, None)

    def restore(self):
        for name, (kind, val) in self._saved.items():
            if kind == "SET":
                if val is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = val
        self._saved.clear()


# ---------------------------------------------------------------------------
# 手动 runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import inspect
    # 接受 0 参或仅有 monkeypatch=None 这样的可选参
    def _is_test_fn(f):
        try:
            params = list(inspect.signature(f).parameters)
        except (TypeError, ValueError):
            return False
        return params == [] or params == ["monkeypatch"]

    fns = [(n, f) for n, f in globals().items()
           if n.startswith("test_") and callable(f) and _is_test_fn(f)]
    passed = 0
    failed = []
    for n, f in fns:
        try:
            f()
            passed += 1
            print(f"  ✓ {n}")
        except AssertionError as e:
            failed.append((n, str(e) or "assertion failed"))
            print(f"  ✗ {n}: {e}")
        except Exception as e:
            failed.append((n, f"{type(e).__name__}: {e}"))
            print(f"  ✗ {n}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if not failed else 1)
