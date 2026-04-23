import os

# 独立路径：config 与 peirastic 包同级，不依赖 deoxys_control
_config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config")
config_root = os.path.abspath(_config_dir)
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

try:
    from peirastic.utils.log_utils import get_peirastic_logger
    get_peirastic_logger()
except Exception:
    pass  # Optional for minimal install

__version__ = "0.1.0"
