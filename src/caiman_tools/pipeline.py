"""CNMF-E pipeline: one input tif movie in, one CaImAn hdf5 result out.

Mirrors the steps of demo_pipeline_cnmfE.ipynb (motion correction -> CNMF-E
source extraction -> component evaluation -> dF/F + deconvolution -> save).

The individual steps are exposed as public functions so they can be run one
at a time (with inspection/tuning in between) from
``notebooks/cnmfe_interactive.ipynb``; ``run_cnmfe_on_movie`` chains them for
the headless ``caiman-cnmfe`` CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as cnmf_params_module

from caiman_tools.view import render_estimates_view


@dataclass
class MovieResult:
    tif_path: Path
    output_path: Path
    n_total: int
    n_accepted: int
    n_rejected: int
    snr_accepted: np.ndarray
    view_html: Path | None


@dataclass
class MotionCorrectionResult:
    fname_mmap: str
    """C-order memmap of the (motion corrected) movie, input to CNMF-E."""
    border_px: int
    mot_correct: MotionCorrect | None
    """None if motion correction was disabled."""


def output_path_for(tif_path: Path) -> Path:
    return tif_path.parent / f"{tif_path.stem}_caiman.hdf5"


TIMESTAMP_COLUMN = "Frame Timestamp (s)"


def timestamps_path_for(tif_path: Path) -> Path:
    return tif_path.parent / f"{tif_path.stem}_timestamps.csv"


def frame_rate_from_timestamps(timestamps_csv: Path) -> float:
    """Frame rate (Hz) from the median inter-frame interval of a timestamps CSV.

    Multi-plane recordings are interleaved, so this is the per-plane rate
    (e.g. 30 Hz acquisition with 3 planes -> 10 Hz).
    """
    with open(timestamps_csv) as f:
        header = [h.strip() for h in f.readline().split(",")]
    col = header.index(TIMESTAMP_COLUMN)
    t = np.loadtxt(timestamps_csv, delimiter=",", skiprows=1, usecols=col, ndmin=1)
    if t.size < 2:
        raise ValueError(f"{timestamps_csv}: need at least 2 timestamps")
    return round(float(1.0 / np.median(np.diff(t))), 2)


def resolve_frame_rate(tif_path: Path, params_fr: float, fr_override: float | None = None) -> float:
    """Frame rate to use for ``tif_path``.

    Priority: explicit override (``--fr``) > ``<stem>_timestamps.csv`` next to the tif >
    ``fr`` from the params.
    """
    if fr_override is not None:
        logger.info("{}: fr={:g} Hz (from --fr)", tif_path.name, fr_override)
        return float(fr_override)
    ts = timestamps_path_for(tif_path)
    if ts.exists():
        try:
            fr = frame_rate_from_timestamps(ts)
        except (ValueError, OSError) as e:
            logger.warning("{}: could not read frame rate from {} ({}); using fr={:g} Hz from params",
                           tif_path.name, ts.name, e, params_fr)
            return float(params_fr)
        logger.info("{}: fr={:g} Hz (from {})", tif_path.name, fr, ts.name)
        return fr
    logger.warning("{}: no {} found; using fr={:g} Hz from params", tif_path.name, ts.name, params_fr)
    return float(params_fr)


def build_cnmf_params(tif_path: Path, cnmf_params: dict[str, Any]):
    """Create a CNMFParams object from the 'cnmf_params' section of the params JSON.

    Returns ``(parameters, motion_correct_flag)``; ``motion_correct`` is a
    caiman_tools-only key, not a CaImAn parameter, so it's split off here.
    """
    cnmf_params = dict(cnmf_params)
    motion_correct = cnmf_params.pop("motion_correct")
    parameters = cnmf_params_module.CNMFParams(
        params_dict={**cnmf_params, "fnames": [str(tif_path)]}
    )
    return parameters, motion_correct


def motion_correct_movie(
    tif_path: Path, parameters, cluster, motion_correct: bool = True
) -> MotionCorrectionResult:
    """Run (rigid or pw-rigid) motion correction and memory-map the result in C order.

    With ``motion_correct=False`` the raw movie is just memory-mapped.
    """
    if not motion_correct:
        fname_new = cm.save_memmap(
            [str(tif_path)], base_name=f"memmap_{tif_path.stem}_", order="C", border_to_0=0, dview=cluster
        )
        return MotionCorrectionResult(fname_new, 0, None)

    motion_group = parameters.get_group("motion")
    mot_correct = MotionCorrect(str(tif_path), dview=cluster, **motion_group)
    mot_correct.motion_correct(save_movie=True)

    pw_rigid = motion_group["pw_rigid"]
    fname_mc = mot_correct.fname_tot_els if pw_rigid else mot_correct.fname_tot_rig
    if pw_rigid:
        bord_px = np.ceil(
            np.maximum(
                np.max(np.abs(mot_correct.x_shifts_els)),
                np.max(np.abs(mot_correct.y_shifts_els)),
            )
        ).astype(int)
    else:
        bord_px = np.ceil(np.max(np.abs(mot_correct.shifts_rig))).astype(int)

    bord_px = 0 if motion_group["border_nan"] == "copy" else int(bord_px)
    fname_new = cm.save_memmap(fname_mc, base_name=f"memmap_{tif_path.stem}_", order="C", border_to_0=bord_px)
    return MotionCorrectionResult(fname_new, bord_px, mot_correct)


def load_images(fname_mmap: str) -> np.ndarray:
    """Load a C-order memmap as a (T, d1, d2) array (no copy)."""
    Yr, dims, T = cm.load_memmap(fname_mmap)
    return Yr.T.reshape((T,) + dims, order="F")


def correlation_pnr(images: np.ndarray, gSig: int) -> tuple[np.ndarray, np.ndarray]:
    """Correlation and peak-to-noise-ratio images (on <=~1000 subsampled frames)."""
    T = images.shape[0]
    return cm.summary_images.correlation_pnr(
        images[:: max(T // 1000, 1)], gSig=gSig, swap_dim=False
    )


def fit_cnmfe(images: np.ndarray, parameters, cluster, n_processes: int, border_px: int = 0):
    """Run CNMF-E source extraction; returns the fitted ``cnmf.CNMF`` model."""
    parameters.change_params(params_dict={"border_pix": int(border_px)})
    model = cnmf.CNMF(n_processes=n_processes, dview=cluster, params=parameters)
    model.fit(images)
    return model


def evaluate_components(model, images: np.ndarray, cluster) -> None:
    """Split components into accepted/rejected using model.params' quality thresholds."""
    model.estimates.evaluate_components(images, model.params, dview=cluster)


def summarize_components(model, name: str = "") -> tuple[int, int, int, np.ndarray]:
    """Log and return (n_total, n_accepted, n_rejected, snr_accepted)."""
    est = model.estimates
    idx_accepted = est.idx_components
    idx_rejected = est.idx_components_bad
    n_total = len(est.C)
    snr_accepted = est.SNR_comp[idx_accepted]

    if snr_accepted.size:
        snr_summary = (
            f"SNR(accepted) min={snr_accepted.min():.2f} "
            f"median={float(np.median(snr_accepted)):.2f} max={snr_accepted.max():.2f}"
        )
    else:
        snr_summary = "SNR(accepted): n/a (0 accepted)"

    logger.info(
        "{}{} accepted / {} rejected / {} total | {}",
        f"{name}: " if name else "",
        len(idx_accepted),
        len(idx_rejected),
        n_total,
        snr_summary,
    )
    return n_total, len(idx_accepted), len(idx_rejected), snr_accepted


def compute_dff_and_deconvolve(model, detrend_df_f_params: dict[str, Any]) -> None:
    """Compute detrended dF/F (F_dff) and deconvolve it (S_dff).

    Skipped if there are no components (CaImAn's deconvolve raises an IndexError then);
    the empty result is still saved, so "0 cells found" is distinguishable from a failure.
    """
    if model.estimates.C is None or len(model.estimates.C) == 0:
        logger.warning("No components found; skipping dF/F and deconvolution")
        return
    if model.estimates.F_dff is None:
        model.estimates.detrend_df_f(**detrend_df_f_params)
    model.estimates.deconvolve(model.params, dff_flag=True)


def save_model(model, output_path: Path, correlation_image: np.ndarray | None = None) -> Path:
    if correlation_image is not None:
        model.estimates.Cn = correlation_image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    return output_path


def cleanup_memmap_files(mc: MotionCorrectionResult) -> None:
    """Delete the intermediate .mmap files of a motion correction run."""
    paths = [mc.fname_mmap]
    if mc.mot_correct is not None:
        paths.append(getattr(mc.mot_correct, "fname_tot_rig", None))
        paths.append(getattr(mc.mot_correct, "fname_tot_els", None))
    for p in paths:
        if p is None:
            continue
        for candidate in p if isinstance(p, list) else [p]:
            try:
                Path(candidate).unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove intermediate memmap file {}", candidate)


def run_cnmfe_on_movie(
    tif_path: Path,
    cnmf_params: dict[str, Any],
    pipeline_params: dict[str, Any],
    cluster,
    n_processes: int,
    output_path: Path,
    cleanup_memmap: bool = False,
    skip_view: bool = False,
) -> MovieResult:
    """Run the full CNMF-E pipeline on a single tif movie and save results to output_path."""
    parameters, motion_correct = build_cnmf_params(tif_path, cnmf_params)
    mc = motion_correct_movie(tif_path, parameters, cluster, motion_correct)
    images = load_images(mc.fname_mmap)

    gSig = parameters.get_group("init")["gSig"]
    correlation_image, _pnr_image = correlation_pnr(images, gSig[0])

    model = fit_cnmfe(images, parameters, cluster, n_processes, mc.border_px)
    evaluate_components(model, images, cluster)
    n_total, n_accepted, n_rejected, snr_accepted = summarize_components(model, tif_path.name)

    compute_dff_and_deconvolve(model, pipeline_params["detrend_df_f"])
    save_model(model, output_path, correlation_image)

    view_html = None
    if not skip_view:
        view_html = output_path.parent / f"{output_path.stem}_view.html"
        try:
            render_estimates_view(
                model.estimates,
                which="accepted",
                output_html=view_html,
                title=f"{output_path.name} (accepted)",
            )
        except ValueError:
            logger.warning("No accepted components in {}; skipping viewer HTML", tif_path.name)
            view_html = None

    if cleanup_memmap:
        cleanup_memmap_files(mc)

    return MovieResult(
        tif_path=tif_path,
        output_path=output_path,
        n_total=n_total,
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        snr_accepted=snr_accepted,
        view_html=view_html,
    )
