from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str = "configs/default_config.yaml") -> Dict[str, Any]:
    """Safely load and parse YAML configuration file.

    Args:
        config_path: Relative or absolute path to the YAML file.

    Returns:
        Dictionary containing configuration parameters.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config
