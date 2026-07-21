"""Oil-Gas VLM Agent Pipeline.

    from pipeline import Pipeline, VLMClient
    p = Pipeline()
    out = p.run_all(seismic_image=img1, log_image=img2)
    print(out.to_dict())
"""
from .adapter import RunPackage, PackageImage, load_run
from .agents import AgentResult, LoopAgent, SingleShotAgent
from .orchestrator import Pipeline, PipelineOutput
from .vlm import VLMClient, VLMResponse, extract_json
from . import adapter, downstream, prompts, tasks, io, exporter

__all__ = [
    "Pipeline", "PipelineOutput",
    "AgentResult", "LoopAgent", "SingleShotAgent",
    "VLMClient", "VLMResponse", "extract_json",
    "RunPackage", "PackageImage", "load_run",
    "adapter", "downstream", "prompts", "tasks", "io", "exporter",
]
