"""loguru-based logging setup.

CaImAn itself logs through the stdlib `logging` module (logger name "caiman"),
and some of its dependencies (scipy, numpy) raise plain Python `warnings`
(UserWarning, RuntimeWarning, ...) straight to stderr. We intercept both and
funnel them into loguru, so everything goes through one consistently-formatted
sink instead of three different output paths. caiman_tools' own status/summary
messages are always shown; everything else is off by default (it's very
verbose even at WARNING) and only shown if `--log-level` is set to something
other than "NONE".
"""

from __future__ import annotations

import inspect
import logging
import sys

from loguru import logger

_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan> - <level>{message}</level>"
)


class _InterceptHandler(logging.Handler):
    """Forwards stdlib `logging` records (from CaImAn) into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(caiman_log_level: str) -> None:
    logger.remove()

    # caiman_tools' own status/summary output: always shown.
    logger.add(
        sys.stderr,
        level="INFO",
        format=_FORMAT,
        filter=lambda record: record["name"].startswith("caiman_tools"),
    )
    # CaImAn's own internal logging: off by default ("NONE"), otherwise
    # shown at the requested verbosity.
    if caiman_log_level != "NONE":
        logger.add(
            sys.stderr,
            level=caiman_log_level,
            format=_FORMAT,
            filter=lambda record: not record["name"].startswith("caiman_tools"),
        )

    # Route plain `warnings.warn()` calls (scipy, numpy, ...) through `logging`
    # too, so they're caught by the same intercept/filtering as everything else.
    logging.captureWarnings(True)

    for logger_name in ("caiman", "py.warnings"):
        std_logger = logging.getLogger(logger_name)
        std_logger.handlers = [_InterceptHandler()]
        std_logger.setLevel(logging.DEBUG)
        std_logger.propagate = False
