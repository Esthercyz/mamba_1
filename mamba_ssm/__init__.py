__version__ = "2.2.6.post3"

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except ImportError:  # pragma: no cover - optional dependency (e.g. Triton) may be missing.
    selective_scan_fn = None  # type: ignore[assignment]
    mamba_inner_fn = None  # type: ignore[assignment]

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:  # pragma: no cover
    Mamba = None  # type: ignore[assignment]

try:
    from mamba_ssm.modules.mamba2 import Mamba2
except ImportError:  # pragma: no cover
    Mamba2 = None  # type: ignore[assignment]

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError:  # pragma: no cover
    MambaLMHeadModel = None  # type: ignore[assignment]

__all__ = [
    "Mamba",
    "Mamba2",
    "MambaLMHeadModel",
    "mamba_inner_fn",
    "selective_scan_fn",
]
