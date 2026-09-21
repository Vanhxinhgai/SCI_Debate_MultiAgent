"""Module to load YAML configuration files for the Scientific Debate Simulator."""
import os
import yaml

DEFAULT_CONFIG = {
    "llm_backends": {
        "mode": "Same model (all agents)",
        "same_model": {
            "backend": "Ollama (local)",
            "model": "qwen2.5:3b",
            "temperature": 0.0,
        },
        "heter_mad": {
            "pro": {
                "backend": "Ollama (local)",
                "model": "qwen2.5:3b",
                "temperature": 0.6,
            },
            "con": {
                "backend": "Ollama (local)",
                "model": "qwen2.5:3b",
                "temperature": 0.8,
            },
            "judge": {
                "backend": "Ollama (local)",
                "model": "qwen2.5:3b",
                "temperature": 0.0,
            }
        }
    },
    "debate_settings": {
        "max_rounds": 2,
        "max_tokens": 300,
        "parallel_opening": False,
    },
    "dar_settings": {
        "enable_dar": False,
        "filter_backend": "Groq (cloud)",
        "filter_model": "llama-3.1-8b-instant",
        "filter_temperature": 0.0,
    },
    "uncertainty_settings": {
        "compute_consensus": False,
        "n_samples": 5,
        "enable_early_stopping": False,
        "use_logprobs": True,
        "uncertainty_backend_mode": "Use same models as debate",
        "uncertainty_groq_model": "llama-3.1-8b-instant",
        "uncertainty_model": "qwen2.5:3b",
    },
    "rag_settings": {
        "enable_rag": False,
        "source": "hybrid",
        "max_results": 3,
    }
}

def merge_dicts(default_dict: dict, override_dict: dict) -> dict:
    """Recursively merges override_dict into default_dict."""
    result = default_dict.copy()
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result

def load_config(config_path="configs/debate_config.yaml") -> dict:
    """Loads configuration from a YAML file. Falls back to default config if file does not exist."""
    if not os.path.exists(config_path):
        return DEFAULT_CONFIG
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
            
        # Recursive dictionary merge to ensure all keys exist
        merged_config = merge_dicts(DEFAULT_CONFIG, user_config)
        return merged_config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}. Using defaults.")
        return DEFAULT_CONFIG
