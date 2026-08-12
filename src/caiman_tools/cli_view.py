"""caiman-view: interactive bokeh viewer for a single CaImAn hdf5 result.

Writes a standalone HTML file (like Estimates.nb_view_components, but for a
plain CLI instead of a Jupyter notebook) letting you step through components
with a slider, seeing the trace, contour, and SNR/spatial-correlation metrics.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

from loguru import logger

from caiman_tools import __version__
from caiman_tools.logging_utils import setup_logging
from caiman_tools.view import view_components


def _open_in_browser(path: Path) -> None:
    """Best-effort, non-blocking browser open. Never lets a hung/slow browser
    launcher (xdg-open, dbus, ...) block or hang this CLI's exit."""
    uri = path.resolve().as_uri()
    try:
        subprocess.Popen(
            ["xdg-open", uri],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        webbrowser.open_new_tab(uri)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactively browse the components in a CaImAn hdf5 result "
        "(writes a standalone bokeh HTML viewer, like estimates.nb_view_components)."
    )
    parser.add_argument("hdf5", type=Path, help="CaImAn hdf5 result file (e.g. from caiman-cnmfe).")
    parser.add_argument("--version", action="version", version=f"caiman-view {__version__}")
    parser.add_argument(
        "--which",
        choices=["accepted", "rejected", "all"],
        default="accepted",
        help="Which components to display (default: accepted).",
    )
    parser.add_argument(
        "--cmap", default="viridis", help="Colormap for the background image (default: viridis)."
    )
    parser.add_argument(
        "--thr", type=float, default=0.9, help="Contour threshold, 0-1 (default: 0.9)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: '<hdf5_stem>_view.html' next to the input).",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Try to open the viewer in a browser after writing it "
        "(off by default: browser auto-launch can hang on headless/remote setups).",
    )
    parser.add_argument(
        "--log-level",
        default="NONE",
        choices=["NONE", "DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity of CaImAn's own internal logging (default: NONE, i.e. hidden).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.log_level)

    hdf5_path = args.hdf5.resolve()
    try:
        output_html = view_components(
            hdf5_path=hdf5_path,
            which=args.which,
            cmap=args.cmap,
            thr=args.thr,
            output_html=args.output.resolve() if args.output else None,
        )
    except Exception:
        logger.exception("Failed to render viewer for {}", hdf5_path)
        return 1

    if args.open_browser:
        _open_in_browser(output_html)

    print(output_html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
