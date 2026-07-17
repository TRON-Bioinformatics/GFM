from importlib.metadata import PackageNotFoundError, version

from .gfm import GFM

try:
    __version__ = version("gfm")
except PackageNotFoundError:
    # Fallback when package metadata is unavailable (e.g., local source usage)
    __version__ = "0.0.0"

__all__ = ["GFM", "__version__"]
