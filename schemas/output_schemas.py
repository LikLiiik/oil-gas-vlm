"""
各Agent输出 JSON Schema 定义（用于校验VLM输出）

每个 schema 定义：
1. 字段名、类型、是否必填
2. 取值范围约束
3. 中文描述（便于理解）
"""

# ============================================================
# Agent 1: SeismicInterpAgent 输出 Schema
# ============================================================

SEISMIC_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["faults", "horizons", "seismic_facies", "summary"],
    "properties": {
        "faults": {
            "type": "array",
            "description": "断层列表",
            "items": {
                "type": "object",
                "required": ["id", "type", "positions", "confidence"],
                "properties": {
                    "id":           {"type": "string", "pattern": "^F\\d+$"},
                    "type":         {"type": "string", "enum": ["normal", "reverse", "strike-slip"]},
                    "dip_direction":{"type": "string", "enum": ["NE", "NW", "SE", "SW", "N", "S", "E", "W"]},
                    "positions":    {"type": "array", "minItems": 2, "maxItems": 10,
                                     "items": {"type": "array", "minItems": 2, "maxItems": 2,
                                               "items": {"type": "number"}}},
                    "throw_ms":     {"type": "number", "minimum": 0},
                    "confidence":   {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "evidence":     {"type": "string"},
                }
            }
        },
        "horizons": {
            "type": "array",
            "description": "层位列表",
            "items": {
                "type": "object",
                "required": ["id", "name", "depth_range_ms", "confidence"],
                "properties": {
                    "id":             {"type": "string"},
                    "name":           {"type": "string"},
                    "type":           {"type": "string",
                                       "enum": ["unconformity", "sequence_boundary",
                                                "flooding_surface", "marker"]},
                    "depth_range_ms": {"type": "array", "minItems": 2, "maxItems": 2,
                                       "items": {"type": "number"}},
                    "amplitude":      {"type": "string", "enum": ["strong", "medium", "weak"]},
                    "continuity":     {"type": "string", "enum": ["good", "moderate", "poor"]},
                    "confidence":     {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "note":           {"type": "string"},
                }
            }
        },
        "seismic_facies": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type", "geological_interpretation"],
                "properties": {
                    "id":                        {"type": "string"},
                    "type":                      {"type": "string",
                                                  "enum": ["parallel", "sub_parallel", "sigmoid",
                                                           "oblique", "chaotic", "mounded",
                                                           "transparent"]},
                    "region_description":        {"type": "string"},
                    "geological_interpretation": {"type": "string"},
                    "confidence":                {"type": "number", "minimum": 0.0, "maximum": 1.0},
                }
            }
        },
        "anomalies": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type", "position", "confidence"],
                "properties": {
                    "id":          {"type": "string"},
                    "type":        {"type": "string",
                                    "enum": ["bright_spot", "flat_spot", "dim_spot",
                                             "gas_chimney", "velocity_pushdown", "pull_up"]},
                    "position":    {"type": "array", "minItems": 2, "maxItems": 2,
                                    "items": {"type": "number"}},
                    "depth_ms":    {"type": "number"},
                    "confidence":  {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "description": {"type": "string"},
                }
            }
        },
        "structural_traps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type", "center", "closure_ms"],
                "properties": {
                    "id":          {"type": "string"},
                    "type":        {"type": "string",
                                    "enum": ["anticline", "fault_block", "fault_nose",
                                             "stratigraphic", "combination"]},
                    "center":      {"type": "array", "minItems": 2, "maxItems": 2,
                                    "items": {"type": "number"}},
                    "closure_ms":  {"type": "number", "minimum": 0},
                    "confidence":  {"type": "number", "minimum": 0.0, "maximum": 1.0},
                }
            }
        },
        "summary": {"type": "string", "description": "综合分析摘要"},
    }
}


# ============================================================
# Agent 2: LogAnalysisAgent 输出 Schema
# ============================================================

LOG_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["lithology_zones", "reservoir_zones", "fluid_zones", "summary"],
    "properties": {
        "well_info": {
            "type": "object",
            "properties": {
                "well_name":    {"type": "string"},
                "depth_range":  {"type": "object",
                                 "properties": {
                                     "top": {"type": "number"}, "bottom": {"type": "number"},
                                     "unit": {"type": "string", "enum": ["m", "ft"]}
                                 }},
                "curves_identified": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["GR", "SP", "RT", "AC", "DEN", "CNL"]}
                },
            }
        },
        "lithology_zones": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "depth_top", "depth_bottom", "lithology", "confidence"],
                "properties": {
                    "id":           {"type": "string", "pattern": "^L\\d+$"},
                    "depth_top":    {"type": "number"},
                    "depth_bottom": {"type": "number"},
                    "lithology":    {"type": "string",
                                     "enum": ["sandstone", "silty_sandstone", "shale",
                                              "limestone", "dolomite", "coal", "salt",
                                              "conglomerate", "mudstone", "anhydrite"]},
                    "gr_range":     {"type": "array", "minItems": 2, "maxItems": 2,
                                     "items": {"type": "number"}},
                    "confidence":   {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "description":  {"type": "string"},
                }
            }
        },
        "reservoir_zones": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "depth_top", "depth_bottom", "porosity_avg_pct"],
                "properties": {
                    "id":                    {"type": "string", "pattern": "^R\\d+$"},
                    "depth_top":             {"type": "number"},
                    "depth_bottom":          {"type": "number"},
                    "net_thickness":         {"type": "number", "minimum": 0},
                    "porosity_avg_pct":      {"type": "number", "minimum": 0, "maximum": 50},
                    "porosity_range_pct":    {"type": "array", "minItems": 2, "maxItems": 2,
                                              "items": {"type": "number"}},
                    "permeability_indication":{"type": "string",
                                               "enum": ["good", "moderate", "poor"]},
                    "confidence":            {"type": "number", "minimum": 0.0, "maximum": 1.0},
                }
            }
        },
        "fluid_zones": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "depth_top", "depth_bottom", "fluid_type", "confidence"],
                "properties": {
                    "id":         {"type": "string", "pattern": "^F\\d+$"},
                    "depth_top":  {"type": "number"},
                    "depth_bottom":{"type": "number"},
                    "fluid_type": {"type": "string",
                                   "enum": ["gas", "oil", "water", "oil_water", "gas_water"]},
                    "resistivity_ohmm":          {"type": "number", "minimum": 0},
                    "water_saturation_estimated_pct": {"type": "number", "minimum": 0, "maximum": 100},
                    "evidence":    {"type": "array", "items": {"type": "string"}},
                    "confidence":  {"type": "number", "minimum": 0.0, "maximum": 1.0},
                }
            }
        },
        "sedimentary_cycles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":            {"type": "string"},
                    "depth_range":   {"type": "array", "minItems": 2, "maxItems": 2},
                    "pattern":       {"type": "string",
                                      "enum": ["fining_upward", "coarsening_upward",
                                               "box_car", "irregular"]},
                    "interpretation":{"type": "string"},
                }
            }
        },
        "key_surfaces": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "depth", "type"],
                "properties": {
                    "id":             {"type": "string"},
                    "depth":          {"type": "number"},
                    "name":           {"type": "string"},
                    "type":           {"type": "string",
                                       "enum": ["sequence_boundary", "flooding_surface",
                                                "unconformity", "lithology_boundary"]},
                    "log_character":  {"type": "string"},
                }
            }
        },
        "summary": {"type": "string"},
    }
}


# ============================================================
# Agent 3: WellSeismicFusionAgent 输出 Schema
# ============================================================

FUSION_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["well_seismic_calibration", "key_geological_interfaces",
                 "seismic_log_correlation", "fusion_summary"],
    "properties": {
        "well_seismic_calibration": {
            "type": "object",
            "required": ["well_name", "correlation_coefficient", "calibration_quality"],
            "properties": {
                "well_name":              {"type": "string"},
                "correlation_coefficient": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                "time_shift_ms":          {"type": "number"},
                "wavelet_phase":          {"type": "string",
                                           "enum": ["zero", "minimum", "mixed"]},
                "wavelet_frequency_hz":   {"type": "number", "minimum": 0},
                "calibration_quality":    {"type": "string",
                                           "enum": ["excellent", "good", "acceptable", "poor"]},
                "issue":                  {"type": "string"},
            }
        },
        "time_depth_table": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["time_ms", "depth_m"],
                "properties": {
                    "time_ms":     {"type": "number", "minimum": 0},
                    "depth_m":     {"type": "number", "minimum": 0},
                    "velocity_ms": {"type": "number", "minimum": 0},
                }
            }
        },
        "key_geological_interfaces": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "depth_m", "time_ms"],
                "properties": {
                    "id":                   {"type": "string"},
                    "name":                 {"type": "string"},
                    "depth_m":              {"type": "number"},
                    "time_ms":              {"type": "number"},
                    "seismic_character":    {"type": "string"},
                    "log_character":        {"type": "string"},
                    "reflection_polarity":  {"type": "string", "enum": ["positive", "negative"]},
                    "amplitude_class":      {"type": "string",
                                             "enum": ["strong", "medium", "weak"]},
                    "lateral_continuity":   {"type": "string",
                                             "enum": ["good", "moderate", "poor"]},
                }
            }
        },
        "seismic_log_correlation": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["interval_name", "correlation_interpretation"],
                "properties": {
                    "interval_name":              {"type": "string"},
                    "seismic_time_range_ms":       {"type": "array", "minItems": 2, "maxItems": 2},
                    "seismic_attributes":          {"type": "object"},
                    "log_characteristics":         {"type": "object"},
                    "correlation_interpretation":  {"type": "string"},
                }
            }
        },
        "cross_well_comparison": {"type": "array"},
        "fusion_summary": {"type": "string"},
    }
}


# ============================================================
# Agent 4: ProspectEvaluationAgent 输出 Schema
# ============================================================

PROSPECT_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["targets", "risk_summary", "overall_summary"],
    "properties": {
        "evaluation_context": {
            "type": "object",
            "properties": {
                "input_sources":   {"type": "array", "items": {"type": "string"}},
                "evaluation_date": {"type": "string"},
            }
        },
        "targets": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "priority_rank",
                             "risk_assessment", "decision"],
                "properties": {
                    "id":              {"type": "string"},
                    "name":            {"type": "string"},
                    "priority_rank":   {"type": "integer", "minimum": 1},
                    "type":            {"type": "string",
                                        "enum": ["structural", "stratigraphic", "combination"]},
                    "location":        {"type": "object"},
                    "trap":            {"type": "object"},
                    "reservoir":       {"type": "object"},
                    "seal":            {"type": "object"},
                    "charge":          {"type": "object"},
                    "risk_assessment": {
                        "type": "object",
                        "required": ["trap_risk", "reservoir_risk", "seal_risk",
                                     "charge_risk", "geological_success_probability_pct"],
                        "properties": {
                            "trap_risk":       {"type": "integer", "minimum": 1, "maximum": 5},
                            "reservoir_risk":  {"type": "integer", "minimum": 1, "maximum": 5},
                            "seal_risk":       {"type": "integer", "minimum": 1, "maximum": 5},
                            "charge_risk":     {"type": "integer", "minimum": 1, "maximum": 5},
                            "overall_risk_score": {"type": "number"},
                            "geological_success_probability_pct": {
                                "type": "number", "minimum": 0, "maximum": 100
                            },
                        }
                    },
                    "resource_estimation": {
                        "type": "object",
                        "properties": {
                            "in_place_mmboe":     {"type": "number", "minimum": 0},
                            "recoverable_mmboe":  {"type": "number", "minimum": 0},
                            "recovery_factor_pct":{"type": "number"},
                        }
                    },
                    "decision": {
                        "type": "object",
                        "required": ["category", "rationale"],
                        "properties": {
                            "category":    {"type": "string",
                                            "enum": ["drill_ready", "data_gap",
                                                     "inventory", "drop"]},
                            "rationale":   {"type": "string"},
                            "next_steps":  {"type": "array", "items": {"type": "string"}},
                        }
                    },
                    "supporting_evidence": {
                        "type": "array", "items": {"type": "string"}
                    },
                }
            }
        },
        "risk_summary": {
            "type": "object",
            "required": ["total_prospects"],
            "properties": {
                "total_prospects":{"type": "integer"},
                "drill_ready":     {"type": "integer"},
                "data_gap":        {"type": "integer"},
                "inventory":       {"type": "integer"},
                "drop":            {"type": "integer"},
            }
        },
        "recommended_workflow": {"type": "array"},
        "overall_summary":      {"type": "string"},
    }
}


# ============================================================
# Workflow Loop: 规划输出 Schema (test_loop.py 的 Phase 1 输出)
# ============================================================

_YOLO_CATEGORY_ITEM = {
    "type": "object",
    "required": ["class_name"],
    "properties": {
        "class_name":             {"type": "string", "minLength": 1},
        "description":            {"type": "string"},
        "expected_cdp_range":     {"type": "array", "minItems": 2, "maxItems": 2},
        "expected_time_range_ms": {"type": "array", "minItems": 2, "maxItems": 2},
        "confidence_threshold":   {"type": "number", "minimum": 0, "maximum": 1},
        "max_detections":         {"type": "integer", "minimum": 1},
    },
}

_TRADITIONAL_RULE_ITEM = {
    "type": "object",
    "required": ["class_name", "rule"],
    "properties": {
        "class_name":            {"type": "string", "minLength": 1},
        "rule":                  {"type": "string", "minLength": 1},
        "expected_depth_ranges": {"type": "array"},
    },
}

WORKFLOW_PLAN_SCHEMA = {
    "type": "object",
    "required": ["scene_understanding", "workflow_steps"],
    "properties": {
        "scene_understanding": {"type": "string", "minLength": 1},
        "workflow_steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["step", "model", "instruction"],
                "properties": {
                    "step":        {"type": "integer", "minimum": 1},
                    "model":       {"type": "string",
                                    "enum": ["sam",
                                             "traditional_code", "seismic_domain_model",
                                             "attribute_extractor", "horizon_tracker",
                                             "facies_classifier", "well_log_analyzer",
                                             "seismic_foundation",
                                             "cig_fault", "cig_channel",
                                             "well_log_ml"]},
                    "reason":      {"type": "string"},
                    "image_name":  {"type": "string"},
                    "instruction": {"type": "object"},
                },
                # 按 model 强制 instruction 必需字段
                "allOf": [
                    {"if": {"properties": {"model": {"const": "traditional_code"}}},
                     "then": {"properties": {"instruction": {
                         "type": "object",
                         "required": ["rules"],
                         "properties": {
                             "rules": {"type": "array", "minItems": 1,
                                       "items": _TRADITIONAL_RULE_ITEM},
                         },
                     }}}},
                    {"if": {"properties": {"model": {"const": "sam"}}},
                     "then": {"properties": {"instruction": {
                         "type": "object",
                         "required": ["prompt_type", "prompt_value", "label"],
                     }}}},
                    {"if": {"properties": {"model": {"const": "seismic_domain_model"}}},
                     "then": {"properties": {"instruction": {
                         "type": "object",
                         "required": ["task"],
                         "properties": {
                             "task": {"type": "string",
                                      "enum": ["fault_detection", "facies_classification",
                                               "fracture"]},
                             "attribute": {"type": "string",
                                           "enum": ["coherence", "structure_tensor",
                                                    "gradient", "variance", "both"]},
                             "confidence_threshold": {"type": "number"},
                             "min_region_area_pixels": {"type": "integer"},
                             "regions_of_interest": {"type": "array"},
                         },
                     }}}},
                    {"if": {"properties": {"model": {"const": "attribute_extractor"}}},
                     "then": {"properties": {"instruction": {
                         "type": "object",
                         "required": ["attributes"],
                         "properties": {
                             "attributes": {"type": "array", "minItems": 1,
                                            "items": {"type": "string"}},
                             "regions_of_interest": {"type": "array"},
                             "spectral_bands": {"type": "array"},
                         },
                     }}}},
                    {"if": {"properties": {"model": {"const": "horizon_tracker"}}},
                     "then": {"properties": {"instruction": {
                         "type": "object",
                         "required": ["seed_points", "tracking_mode"],
                         "properties": {
                             "seed_points": {"type": "array", "minItems": 1},
                             "tracking_mode": {"type": "string",
                                               "enum": ["peak", "trough", "correlation",
                                                        "zero_crossing"]},
                             "search_window_samples": {"type": "integer", "minimum": 1},
                             "horizon_name": {"type": "string"},
                         },
                     }}}},
                    {"if": {"properties": {"model": {"const": "facies_classifier"}}},
                     "then": {"properties": {"instruction": {
                         "type": "object",
                         "required": ["n_clusters"],
                         "properties": {
                             "n_clusters": {"type": "integer", "minimum": 2, "maximum": 20},
                             "attribute_list": {"type": "array",
                                                "items": {"type": "string"}},
                             "regions_of_interest": {"type": "array"},
                             "method": {"type": "string",
                                        "enum": ["kmeans", "gmm"]},
                         },
                     }}}},
                    {"if": {"properties": {"model": {"const": "well_log_analyzer"}}},
                     "then": {"properties": {"instruction": {
                         "type": "object",
                         "required": ["analysis_type"],
                         "properties": {
                             "analysis_type": {"type": "string",
                                               "enum": ["curve_segmentation",
                                                        "lithology_classification",
                                                        "fluid_identification",
                                                        "full_analysis"]},
                             "rules": {"type": "array"},
                             "depth_range": {"type": "object",
                                             "properties": {
                                                 "top_m": {"type": "number"},
                                                 "bottom_m": {"type": "number"},
                                             }},
                         },
                     }}}},
                    {"if": {"properties": {"model": {"const": "seismic_foundation"}}},
                     "then": {"properties": {"instruction": {
                         "type": "object",
                         "required": ["task"],
                         "properties": {
                             "task": {"type": "string",
                                      "enum": ["facies_classification",
                                               "feature_extraction"]},
                             "regions_of_interest": {"type": "array"},
                         },
                     }}}},
                    {"if": {"properties": {"model": {"const": "cig_fault"}}},
                     "then": {"properties": {"instruction": {
                         "type": "object",
                         "properties": {
                             "threshold": {"type": "number"},
                             "scale": {"type": "number"},
                         },
                     }}}},
                    {"if": {"properties": {"model": {"const": "cig_channel"}}},
                     "then": {"properties": {"instruction": {
                         "type": "object",
                         "properties": {
                             "threshold": {"type": "number"},
                             "scales": {"type": "array"},
                         },
                     }}}},
                    {"if": {"properties": {"model": {"const": "well_log_ml"}}},
                     "then": {"properties": {"instruction": {
                         "type": "object",
                         "required": ["analysis_type"],
                         "properties": {
                             "analysis_type": {"type": "string",
                                               "enum": ["lithology", "fluid",
                                                        "full"]},
                             "depth_range": {"type": "object"},
                         },
                     }}}},
                ],
            }
        },
        "dependencies":          {"type": "array"},
        "verification_strategy": {"type": "string",
                                  "enum": ["per_step", "batch", "none"]},
        "max_iterations":        {"type": "integer", "minimum": 1, "maximum": 10},
    }
}


# ============================================================
# Workflow Loop: 验证输出 Schema (test_loop.py 的 Phase 3 输出)
# ============================================================

WORKFLOW_VERIFICATION_SCHEMA = {
    "type": "object",
    "required": ["verified", "need_retry"],
    "properties": {
        "verified": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["is_real"],
                "properties": {
                    "step":              {"type": "integer"},
                    "model":             {"type": "string"},
                    "result_id":         {"type": "string"},
                    "bbox_xyxy_norm":    {"type": "array", "minItems": 4, "maxItems": 4,
                                          "items": {"type": "number"}},
                    "is_real":           {"type": "boolean"},
                    "confidence":        {"type": "number",
                                          "minimum": 0.0, "maximum": 1.0},
                    "geological_reason": {"type": "string"},
                    "rejection_reason":  {"type": ["string", "null"]},
                }
            }
        },
        "false_positives":  {"type": "integer", "minimum": 0},
        "missed_targets":   {"type": "array"},
        "need_retry":       {"type": "boolean"},
        "retry_instructions": {"type": ["object", "null"]},
        "final_summary":    {"type": "string"},
    }
}


# ============================================================
# 校验工具
# ============================================================

def validate_output(schema: dict, data: dict) -> tuple[bool, list[str]]:
    """简单的 schema 校验，返回 (是否通过, 错误列表)"""
    import jsonschema
    try:
        jsonschema.validate(data, schema)
        return True, []
    except jsonschema.ValidationError as e:
        # e.message 短，比 str(e) 更适合作为反馈给 VLM 的错误信息
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        return False, [f"{path}: {e.message}"]


# 映射：agent_name → schema
AGENT_SCHEMAS = {
    "seismic_interp":       SEISMIC_OUTPUT_SCHEMA,
    "log_analysis":         LOG_OUTPUT_SCHEMA,
    "well_seismic_fusion":  FUSION_OUTPUT_SCHEMA,
    "prospect_evaluation":  PROSPECT_OUTPUT_SCHEMA,
    "workflow_plan":        WORKFLOW_PLAN_SCHEMA,
    "workflow_verification": WORKFLOW_VERIFICATION_SCHEMA,
}
