try:
    from caiman_tools._version import version as __version__
except ImportError:
    __version__ = "unknown"


def hello() -> str:
    return "Hello from caiman-tools!"
