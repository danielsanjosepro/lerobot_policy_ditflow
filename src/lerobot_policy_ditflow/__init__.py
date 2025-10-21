"""lerobot_policy_ditflow package initialization."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use lerobot_policy_ditflow."
    )

from lerobot_policy_ditflow.configuration_ditflow import DiTFlowConfig
from lerobot_policy_ditflow.modeling_ditflow import DiTFlowPolicy
from lerobot_policy_ditflow.processor_ditflow import make_ditflow_pre_post_processors

__all__ = [
    "DiTFlowConfig",
    "DiTFlowPolicy",
    "make_ditflow_pre_post_processors",
]
