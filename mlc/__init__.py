"""mlc: an ahead-of-time compiler for static-shape PyTorch inference."""

from .api import compile
from .config import Config, ELEMENTWISE_ONLY, NO_FUSION

__version__ = "0.1.0"
__all__ = ["compile", "Config", "ELEMENTWISE_ONLY", "NO_FUSION"]
