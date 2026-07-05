"""M-Ana data-science toolkit.

Subpackages are loaded on demand so optional frameworks such as TensorFlow
and PyTorch are not imported during a basic ``import MAna``.
"""

from importlib import import_module

from ._version import __version__

__all__ = [
    "__version__",
    "analysis",
    "big",
    "data",
    "database",
    "modeling",
    "nlp",
    "rag",
    "recommend",
    "stata",
    "timeseries",
]

_SUBPACKAGES = {name: f"{__name__}.{name}" for name in __all__[1:]}
_LEGACY_EXPORT_PACKAGES = ("analysis", "data", "modeling")


def __getattr__(name):
    if name in _SUBPACKAGES:
        module = import_module(_SUBPACKAGES[name])
        globals()[name] = module
        return module

    # Preserve common legacy access such as ``MAna.read_data`` without
    # eagerly importing every historical module at startup.
    for package_name in _LEGACY_EXPORT_PACKAGES:
        module = import_module(_SUBPACKAGES[package_name])
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
