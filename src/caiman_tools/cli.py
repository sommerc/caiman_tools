"""caiman-cnmfe: batch CNMF-E extraction CLI.

Takes a list of tif movies and an optional JSON params file, and runs the
CNMF-E pipeline from demo_pipeline_cnmfE.ipynb non-interactively on each one,
writing '<tif_stem>_caiman.hdf5' next to each input tif.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from loguru import logger

import caiman as cm

from caiman_tools.logging_utils import setup_logging
from caiman_tools.params import load_params
from caiman_tools.pipeline import MovieResult, run_cnmfe_on_movie


def _output_path_for(tif_path: Path) -> Path:
    return tif_path.parent / f"{tif_path.stem}_caiman.hdf5"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch CNMF-E source extraction over tif movies using CaImAn."
    )
    parser.add_argument("tifs", nargs="+", type=Path, help="Input tif movie(s) to process.")
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="JSON file overriding defaults (only the provided keys are overridden).",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to pipeline.n_processes in params, "
        "or CaImAn's own auto-detected default.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess and overwrite outputs that already exist.",
    )
    parser.add_argument(
        "--cleanup-memmap",
        action="store_true",
        help="Delete intermediate .mmap files after a successful save.",
    )
    parser.add_argument(
        "--skip-view",
        action="store_true",
        help="Don't auto-generate the '<output_stem>_view.html' interactive component "
        "viewer (see caiman-view) after saving each result.",
    )
    parser.add_argument(
        "--log-level",
        default="NONE",
        choices=["NONE", "DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity of CaImAn's own internal logging (default: NONE, i.e. hidden). "
        "caiman_tools' own progress/summary output is always shown.",
    )
    return parser


def _log_batch_summary(results: list[MovieResult]) -> None:
    if not results:
        return
    n_total = sum(r.n_total for r in results)
    n_accepted = sum(r.n_accepted for r in results)
    n_rejected = sum(r.n_rejected for r in results)
    all_snr = np.concatenate([r.snr_accepted for r in results])

    logger.info(
        "Batch summary: {} file(s) | {} accepted / {} rejected / {} total",
        len(results),
        n_accepted,
        n_rejected,
        n_total,
    )
    if all_snr.size:
        logger.info(
            "Batch SNR(accepted): min={:.2f} median={:.2f} max={:.2f}",
            all_snr.min(),
            float(np.median(all_snr)),
            all_snr.max(),
        )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.log_level)

    all_params = load_params(args.params)
    cnmf_params = all_params["cnmf_params"]
    pipeline_params = all_params["pipeline"]

    n_processes = args.n_processes or pipeline_params.get("n_processes")

    logger.info("Starting cluster (n_processes={})", n_processes)
    _, cluster, n_processes = cm.cluster.setup_cluster(
        backend="multiprocessing", n_processes=n_processes, ignore_preexisting=False
    )

    results: list[MovieResult] = []
    failures: list[Path] = []
    try:
        for tif_path in args.tifs:
            tif_path = tif_path.resolve()
            output_path = _output_path_for(tif_path)

            if output_path.exists() and not args.overwrite:
                logger.info(
                    "Skipping {} (output already exists: {})", tif_path.name, output_path.name
                )
                continue

            logger.info("Processing {} -> {}", tif_path.name, output_path.name)
            try:
                result = run_cnmfe_on_movie(
                    tif_path=tif_path,
                    cnmf_params=cnmf_params,
                    pipeline_params=pipeline_params,
                    cluster=cluster,
                    n_processes=n_processes,
                    output_path=output_path,
                    cleanup_memmap=args.cleanup_memmap,
                    skip_view=args.skip_view,
                )
                results.append(result)
                logger.success(
                    "{}: {} accepted / {} rejected -> {}",
                    tif_path.name,
                    result.n_accepted,
                    result.n_rejected,
                    output_path.name,
                )
            except Exception:
                logger.exception("Failed to process {}", tif_path)
                failures.append(tif_path)
    finally:
        cm.stop_server(dview=cluster)

    _log_batch_summary(results)

    if failures:
        print(f"{len(failures)}/{len(args.tifs)} file(s) failed:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
