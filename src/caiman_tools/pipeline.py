"""Headless CNMF-E pipeline: one input tif movie in, one CaImAn hdf5 result out.

Mirrors the interactive steps of demo_pipeline_cnmfE.ipynb (motion correction ->
CNMF-E source extraction -> component evaluation -> dF/F -> save), with all
plotting/inspection cells removed since this runs non-interactively.
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


def _motion_correct(tif_path: Path, parameters, cluster):
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
    fname_new = cm.save_memmap(fname_mc, base_name="memmap_", order="C", border_to_0=bord_px)
    return fname_new, bord_px, mot_correct


def _cleanup_memmap_files(*paths: Any) -> None:
    for p in paths:
        if p is None:
            continue
        candidates = p if isinstance(p, list) else [p]
        for candidate in candidates:
            try:
                Path(candidate).unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove intermediate memmap file %s", candidate)


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
    cnmf_params = dict(cnmf_params)
    motion_correct = cnmf_params.pop("motion_correct")

    parameters = cnmf_params_module.CNMFParams(
        params_dict={**cnmf_params, "fnames": [str(tif_path)]}
    )

    mot_correct = None
    if motion_correct:
        fname_new, bord_px, mot_correct = _motion_correct(tif_path, parameters, cluster)
    else:
        bord_px = 0
        fname_new = cm.save_memmap(
            [str(tif_path)], base_name="memmap_", order="C", border_to_0=0, dview=cluster
        )

    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order="F")

    parameters.change_params(params_dict={"border_pix": int(bord_px)})

    gSig = parameters.get_group("init")["gSig"]
    correlation_image, _pnr_image = cm.summary_images.correlation_pnr(
        images[:: max(T // 1000, 1)], gSig=gSig[0], swap_dim=False
    )

    model = cnmf.CNMF(n_processes=n_processes, dview=cluster, params=parameters)
    model.fit(images)

    model.estimates.evaluate_components(images, model.params, dview=cluster)

    idx_accepted = model.estimates.idx_components
    idx_rejected = model.estimates.idx_components_bad
    n_total = len(model.estimates.C)
    snr_accepted = model.estimates.SNR_comp[idx_accepted]

    if snr_accepted.size:
        snr_summary = (
            f"SNR(accepted) min={snr_accepted.min():.2f} "
            f"median={float(np.median(snr_accepted)):.2f} max={snr_accepted.max():.2f}"
        )
    else:
        snr_summary = "SNR(accepted): n/a (0 accepted)"

    logger.info(
        "{}: {} accepted / {} rejected / {} total | {}",
        tif_path.name,
        len(idx_accepted),
        len(idx_rejected),
        n_total,
        snr_summary,
    )

    if model.estimates.F_dff is None:
        model.estimates.detrend_df_f(**pipeline_params["detrend_df_f"])

    model.estimates.Cn = correlation_image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

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
        extra = []
        if mot_correct is not None:
            extra.append(getattr(mot_correct, "fname_tot_rig", None))
            extra.append(getattr(mot_correct, "fname_tot_els", None))
        _cleanup_memmap_files(fname_new, *extra)

    return MovieResult(
        tif_path=tif_path,
        output_path=output_path,
        n_total=n_total,
        n_accepted=len(idx_accepted),
        n_rejected=len(idx_rejected),
        snr_accepted=snr_accepted,
        view_html=view_html,
    )
