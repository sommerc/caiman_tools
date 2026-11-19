"""Multi-session registration of CNMF-E results (same animal, same day).

Entry point is a session metadata CSV (one row per recording, possibly with several
e-focus planes). Workflow, used by ``notebooks/multisession_registration.ipynb``:

1. ``load_session_table``  - read + clean the metadata CSV
2. ``find_plane_results``  - locate ``*_caiman.hdf5`` + timestamps per (recording, e-focus)
3. ``plane_stats``         - per-plane summary (n accepted, SNR); the e-focus choice itself
                              is done in the notebook
4. ``register_sessions``   - caiman.base.rois.register_multisession on accepted components
                              of a user-selected set of sessions, in temporal order
5. ``save_registered``     - one HDF5 with traces aligned to the union cell index
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.sparse
from loguru import logger

_MOVIE_TIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})-(\d{2})-(\d{2})-(\d{2})")


# --------------------------------------------------------------------------- inventory


def load_session_table(csv_path: Path) -> pd.DataFrame:
    """Read the session metadata CSV; one row per recording.

    Adds: ``recording`` (movie stem), ``animal_id`` (str), ``start_time`` (from the
    movie file name), ``date``, ``efoci`` (list[int]). Rows without a movie file are
    dropped with a warning.
    """
    df = pd.read_csv(csv_path)
    bad = df["movie_file"].isna() | df["output_subdir"].isna()
    if bad.any():
        logger.warning("Dropping {} malformed row(s) without movie_file/output_subdir: {}",
                       int(bad.sum()), [int(i) + 2 for i in df.index[bad]])  # +2: header, 1-based
    df = df[~bad].copy()

    df["recording"] = df["movie_file"].map(lambda s: Path(s).stem)
    df["animal_id"] = df["acq_Animal ID"].map(lambda v: str(int(v)) if pd.notna(v) else "unknown")

    def _start(s: str) -> pd.Timestamp:
        m = _MOVIE_TIME_RE.search(s)
        if m is None:
            raise ValueError(f"Cannot parse recording time from {s!r}")
        return pd.Timestamp(f"{m[1]} {m[2]}:{m[3]}:{m[4]}")

    df["start_time"] = df["movie_file"].map(_start)
    df["date"] = df["start_time"].dt.date.astype(str)
    df["efoci"] = df["efocus_values"].map(lambda s: [int(x) for x in ast.literal_eval(s)])
    return df.sort_values(["animal_id", "start_time"]).reset_index(drop=True)


def find_plane_results(sessions: pd.DataFrame) -> pd.DataFrame:
    """One row per (recording, e-focus) with paths to the CaImAn hdf5 and timestamps.

    Matches both naming schemes produced upstream:
    ``<stem>_DeinterleavedPlane_<ef>_multiplanar_nPP_caiman.hdf5`` and
    ``<stem>_<ef>_uniplanar_nPP_caiman.hdf5``.
    """
    rows = []
    for _, s in sessions.iterrows():
        out_dir = Path(s["output_subdir"])
        for ef in s["efoci"]:
            hits = sorted(out_dir.glob(f"{s['recording']}*_{ef}_*planar*_caiman.hdf5"))
            hdf5 = hits[0] if hits else None
            if len(hits) > 1:
                logger.warning("{} e-focus {}: {} hdf5 candidates, using {}",
                               s["recording"], ef, len(hits), hits[0].name)
            ts = None
            if hdf5 is not None:
                ts = hdf5.with_name(hdf5.name.replace("_caiman.hdf5", "_timestamps.csv"))
                ts = ts if ts.exists() else None
            status = "ok" if hdf5 is not None and ts is not None else (
                "missing hdf5" if hdf5 is None else "missing timestamps")
            rows.append({
                "animal_id": s["animal_id"], "date": s["date"], "start_time": s["start_time"],
                "recording": s["recording"], "efocus": ef, "n_planes": len(s["efoci"]),
                "hdf5": hdf5, "timestamps": ts, "status": status,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- plane scoring


def plane_stats(hdf5_path: Path, timestamps_path: Path | None = None) -> dict[str, Any]:
    """Cheap summary of a CaImAn result (reads only small datasets via h5py)."""
    with h5py.File(hdf5_path, "r") as h:
        est = h["estimates"]
        idx = est["idx_components"][()] if est["idx_components"].shape != () else None
        snr = est["SNR_comp"][()]
        n_total, T = est["C"].shape
        fr = float(h["params/data/fr"][()])
    if idx is None:  # never evaluated -> all components
        idx = np.arange(n_total)
    snr_acc = snr[idx]
    finite = snr_acc[np.isfinite(snr_acc)]
    out = dict(
        n_total=int(n_total),
        n_accepted=int(len(idx)),
        snr_median=float(np.median(finite)) if finite.size else np.nan,
        snr_mean=float(np.mean(finite)) if finite.size else np.nan,
        n_snr_inf=int((~np.isfinite(snr_acc)).sum()),
        T=int(T),
        fr_params=fr,
    )
    if timestamps_path is not None:
        t = pd.read_csv(timestamps_path)["Frame Timestamp (s)"].to_numpy()
        out["fr_timestamps"] = float(1 / np.median(np.diff(t)))
        out["n_timestamps"] = int(len(t))
    return out


def add_plane_stats(planes: pd.DataFrame) -> pd.DataFrame:
    stats = [
        plane_stats(r.hdf5, r.timestamps) if r.status == "ok" else {}
        for r in planes.itertuples()
    ]
    out = pd.concat([planes.reset_index(drop=True), pd.DataFrame(stats)], axis=1)
    out["fr_mismatch"] = (out["status"] == "ok") & ~np.isclose(
        out["fr_params"], out["fr_timestamps"], rtol=0.05)
    return out


# --------------------------------------------------------------------------- registration


@dataclass
class SessionData:
    recording: str
    animal_id: str
    start_time: pd.Timestamp
    hdf5: Path
    timestamps_path: Path
    efocus: int
    fr: float
    idx_accepted: np.ndarray           # original CaImAn component indices
    A: scipy.sparse.csc_matrix         # pixels x accepted
    Cn: np.ndarray
    traces: dict[str, np.ndarray]      # name -> (n_accepted, T)
    timestamps: np.ndarray             # unix seconds, (T,)
    snr: np.ndarray
    r_values: np.ndarray


TRACE_ATTRS = ("F_dff", "C", "S_dff")


def load_session(row) -> SessionData:
    from caiman.source_extraction.cnmf.cnmf import load_CNMF

    est = load_CNMF(str(row.hdf5)).estimates
    idx = np.asarray(est.idx_components if est.idx_components is not None
                     else np.arange(est.A.shape[1]))
    traces = {}
    for name in TRACE_ATTRS:
        arr = getattr(est, name, None)
        if arr is None:
            if len(idx):  # empty results have no traces by design
                logger.warning("{}: estimates.{} missing", row.recording, name)
            continue
        traces[name] = np.asarray(arr)[idx].astype(np.float32)
    t = pd.read_csv(row.timestamps)["Frame Timestamp (s)"].to_numpy()
    T = est.C.shape[1]
    if len(t) != T:
        logger.warning("{}: {} timestamps vs {} frames; truncating to the shorter",
                       row.recording, len(t), T)
        n = min(len(t), T)
        t = t[:n]
        traces = {k: v[:, :n] for k, v in traces.items()}
    return SessionData(
        recording=row.recording, animal_id=str(row.animal_id), start_time=row.start_time, hdf5=Path(row.hdf5),
        timestamps_path=Path(row.timestamps), efocus=int(row.efocus),
        fr=float(1 / np.median(np.diff(t))),
        idx_accepted=idx, A=scipy.sparse.csc_matrix(est.A)[:, idx], Cn=np.asarray(est.Cn),
        traces=traces, timestamps=t, snr=np.asarray(est.SNR_comp)[idx],
        r_values=np.asarray(est.r_values)[idx],
    )


@dataclass
class Registration:
    name: str                          # used for the output file name
    efoci: list[int]                   # e-focus per session (may differ)
    sessions: list[SessionData]
    A_union: scipy.sparse.csc_matrix
    assignments: np.ndarray            # (n_union, n_sessions), accepted-local idx or NaN
    matchings: list[list[int]]
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def dims(self) -> tuple[int, int]:
        return self.sessions[0].Cn.shape

    @property
    def n_present(self) -> np.ndarray:
        return (~np.isnan(self.assignments)).sum(axis=1)

    def summary(self) -> pd.DataFrame:
        n = self.n_present
        return pd.DataFrame(
            {"n_cells": [int((n == k).sum()) for k in range(1, len(self.sessions) + 1)]},
            index=pd.Index(range(1, len(self.sessions) + 1), name="found in n sessions"),
        )

    def aligned_traces(self, name: str = "F_dff") -> list[np.ndarray]:
        """Per session: (n_union, T_s) array; NaN rows where the cell wasn't found."""
        out = []
        for j, s in enumerate(self.sessions):
            arr = np.full((len(self.assignments), s.traces[name].shape[1]), np.nan, np.float32)
            present = ~np.isnan(self.assignments[:, j])
            arr[present] = s.traces[name][self.assignments[present, j].astype(int)]
            out.append(arr)
        return out

    def assignment_table(self) -> pd.DataFrame:
        """union_id x session -> original CaImAn component index (-1 = not found)."""
        cols = {}
        for j, s in enumerate(self.sessions):
            col = np.full(len(self.assignments), -1, int)
            present = ~np.isnan(self.assignments[:, j])
            col[present] = s.idx_accepted[self.assignments[present, j].astype(int)]
            cols[s.recording] = col
        df = pd.DataFrame(cols)
        df.index.name = "union_id"
        df["n_present"] = self.n_present
        return df


def default_name(sessions: list[SessionData]) -> str:
    """``<animal>_<first date>_to_<last date>`` (animals joined with '+' if several)."""
    animals = "+".join(dict.fromkeys(s.animal_id for s in sessions))
    first, last = sessions[0].start_time, sessions[-1].start_time
    return f"{animals}_{first:%Y-%m-%d}_to_{last:%Y-%m-%d}"


def register_sessions(
    sessions: list[SessionData],
    name: str | None = None,
    thresh_cost: float = 0.7,
    max_dist: float = 10,
    max_thr: float = 0,
    align_flag: bool = True,
    use_opt_flow: bool = True,
    enclosed_thr: float | None = None,
) -> Registration:
    """Register the accepted components of ``sessions`` to a union set.

    Sessions are sorted by start time first (register_multisession works
    sequentially: 1 -> 2, union -> 3, ...). Sessions without accepted components
    are dropped with a warning.
    """
    from caiman.base.rois import register_multisession

    sessions = sorted(sessions, key=lambda s: s.start_time)
    empty = [s for s in sessions if s.A.shape[1] == 0]
    for s in empty:
        logger.warning("{}: no accepted components, dropped from registration", s.recording)
    sessions = [s for s in sessions if s.A.shape[1] > 0]
    if len(sessions) < 2:
        raise ValueError(f"Need at least 2 sessions with components, got {len(sessions)}")
    shapes = {s.Cn.shape for s in sessions}
    if len(shapes) > 1:
        detail = ", ".join(f"{s.recording}: {s.Cn.shape}" for s in sessions)
        raise ValueError(
            f"Sessions have different FOV shapes ({detail}). The movies are probably "
            "rotated/transposed relative to each other; fix the orientation of the "
            "tif export and re-run caiman-cnmfe."
        )
    if len({s.animal_id for s in sessions}) > 1:
        logger.warning("Registering sessions of different animals: {}",
                       sorted({s.animal_id for s in sessions}))

    params = dict(thresh_cost=thresh_cost, max_dist=max_dist, max_thr=max_thr,
                  align_flag=align_flag, use_opt_flow=use_opt_flow, enclosed_thr=enclosed_thr)
    dims = sessions[0].Cn.shape
    A_union, assignments, matchings = register_multisession(
        A=[s.A for s in sessions], dims=dims, templates=[s.Cn for s in sessions], **params
    )
    return Registration(
        name=name or default_name(sessions), efoci=[s.efocus for s in sessions], sessions=sessions,
        A_union=scipy.sparse.csc_matrix(A_union), assignments=np.asarray(assignments, float),
        matchings=matchings, params=params,
    )


# --------------------------------------------------------------------------- saving


def output_stem(reg: Registration) -> str:
    return f"{reg.name}_registered"


def save_registered(reg: Registration, out_dir: Path) -> tuple[Path, Path]:
    """Write ``<name>_registered.h5`` (+ assignments CSV).

    HDF5 layout (all trace arrays share the union cell index as first axis)::

        attrs: name, n_sessions, n_union, registration_params (json),
               recordings, animal_ids, start_times, efoci (one entry per session)
        /union/footprints     (n_union, d1, d2) float32   in the LAST session's FOV
        /union/assignments    (n_union, n_sessions) int   orig. CaImAn component idx, -1 = absent
        /union/n_present      (n_union,)
        /sessions/<k>/        k = 0..n-1 in recording order
            attrs: recording, animal_id, start_time, efocus, fr, hdf5
            timestamps        (T,) float64 unix seconds
            F_dff, C, S_dff   (n_union, T) float32, NaN rows where the cell is absent
            Cn                (d1, d2) correlation image (registration template)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / f"{output_stem(reg)}.h5"
    csv_path = out_dir / f"{output_stem(reg)}_assignments.csv"
    d1, d2 = reg.dims
    table = reg.assignment_table()
    comp = dict(compression="gzip", compression_opts=4)

    with h5py.File(h5_path, "w") as h:
        h.attrs.update(
            name=reg.name, efoci=np.asarray(reg.efoci),
            recordings=[s.recording for s in reg.sessions],
            animal_ids=[s.animal_id for s in reg.sessions],
            start_times=[str(s.start_time) for s in reg.sessions],
            n_sessions=len(reg.sessions), n_union=len(reg.assignments),
            registration_params=json.dumps(reg.params),
        )
        u = h.create_group("union")
        fp = reg.A_union.toarray().reshape(d1, d2, -1, order="F").transpose(2, 0, 1)
        u.create_dataset("footprints", data=fp.astype(np.float32), **comp)
        u.create_dataset("assignments", data=table[[s.recording for s in reg.sessions]].to_numpy())
        u.create_dataset("n_present", data=reg.n_present)

        aligned = {name: reg.aligned_traces(name) for name in reg.sessions[0].traces}
        for j, s in enumerate(reg.sessions):
            g = h.create_group(f"sessions/{j}")
            g.attrs.update(recording=s.recording, animal_id=s.animal_id, start_time=str(s.start_time),
                           efocus=s.efocus, fr=s.fr, hdf5=str(s.hdf5))
            g.create_dataset("timestamps", data=s.timestamps)
            g.create_dataset("Cn", data=s.Cn.astype(np.float32))
            for name, arrs in aligned.items():
                g.create_dataset(name, data=arrs[j], **comp)

    table.to_csv(csv_path)
    logger.info("Wrote {} and {}", h5_path.name, csv_path.name)
    return h5_path, csv_path


def load_registered(h5_path: Path) -> dict[str, Any]:
    """Read a file written by ``save_registered`` into plain numpy/dicts."""
    with h5py.File(h5_path, "r") as h:
        out = dict(h.attrs)
        out["union"] = {k: v[()] for k, v in h["union"].items()}
        out["sessions"] = []
        for k in sorted(h["sessions"], key=int):
            g = h["sessions"][k]
            out["sessions"].append({**dict(g.attrs), **{n: d[()] for n, d in g.items()}})
    return out


# --------------------------------------------------------------------------- plotting


def _contours(ax, A, dims, colors, thr: float = 0.5, lw: float = 1.0) -> None:
    """Draw each footprint's contour at ``thr`` x its max."""
    A = scipy.sparse.csc_matrix(A)
    for k in range(A.shape[1]):
        fp = A[:, k].toarray().reshape(dims, order="F")
        if fp.max() > 0:
            ax.contour(fp, levels=[thr * fp.max()], colors=[colors[k]], linewidths=lw)


def plot_registration(reg: Registration, thr: float = 0.5):
    """Per session: template + contours (green = matched in >=2 sessions, red = only here).
    Last panel: union footprints on the last session's template, coloured by n sessions."""
    import matplotlib.pyplot as plt

    n = len(reg.sessions)
    n_present = reg.n_present
    ncols = min(n + 1, 4)
    nrows = -(-(n + 1) // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
                            constrained_layout=True, squeeze=False, sharex=True, sharey=True)
    axs = axs.ravel()
    for ax in axs[n + 1:]:
        ax.axis("off")
    for j, (ax, s) in enumerate(zip(axs, reg.sessions)):
        union_of = np.asarray(reg.matchings[j], int)          # accepted-local -> union id
        colors = ["tab:green" if n_present[u] > 1 else "tab:red" for u in union_of]
        ax.imshow(s.Cn, cmap="gray")
        _contours(ax, s.A, reg.dims, colors, thr)
        ax.set_title(f"{j}: {s.recording[6:25]} ef{s.efocus}\n{s.A.shape[1]} cells, "
                     f"{sum(c == 'tab:green' for c in colors)} matched", fontsize=9)
        ax.axis("off")
    cmap = plt.get_cmap("viridis", n)
    ax_u = axs[n]
    ax_u.imshow(reg.sessions[-1].Cn, cmap="gray")
    _contours(ax_u, reg.A_union, reg.dims, [cmap(k - 1) for k in n_present], thr)
    ax_u.set_title(f"union: {len(n_present)} cells (colour = # sessions)", fontsize=9)
    ax_u.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.5, n + 0.5))
    fig.colorbar(sm, ax=ax_u, ticks=range(1, n + 1), shrink=0.8)
    return fig


def plot_matched_traces(reg: Registration, union_ids, name: str = "F_dff", spacing: float | None = None):
    """Traces of the given union cells, sessions placed back to back in temporal order.

    The x axis is minutes within each session (each starts at 0); sessions are separated
    by a small gap and a dashed line, labelled with their index and start time. The real
    time between sessions is not shown.
    """
    import matplotlib.pyplot as plt

    union_ids = list(union_ids)
    aligned = reg.aligned_traces(name)
    if spacing is None:
        vals = np.concatenate([a[union_ids].ravel() for a in aligned])
        spacing = np.nanpercentile(vals, 99.5) if np.isfinite(vals).any() else 1.0

    durations = [(s.timestamps[-1] - s.timestamps[0]) / 60 for s in reg.sessions]
    gap = 0.02 * sum(durations)
    offsets = np.concatenate([[0], np.cumsum(np.add(durations, gap))[:-1]])

    fig, ax = plt.subplots(figsize=(12, 0.6 * len(union_ids) + 1.8), constrained_layout=True)
    for j, (s, a, off, dur) in enumerate(zip(reg.sessions, aligned, offsets, durations)):
        t = (s.timestamps - s.timestamps[0]) / 60 + off
        for i, u in enumerate(union_ids):
            ax.plot(t, a[u] + i * spacing, lw=0.5, color=f"C{j}")
        if j:
            ax.axvline(off - gap / 2, color="0.5", ls="--", lw=0.8)
        ax.text(off + dur / 2, 1.01, f"{j}: {s.start_time:%Y-%m-%d %H:%M}\nef{s.efocus}",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8,
                color=f"C{j}")

    # ticks: start and end of each session, in minutes within that session
    ticks = np.concatenate([[off, off + dur] for off, dur in zip(offsets, durations)])
    ax.set_xticks(ticks, [f"{v:.0f}" for d in durations for v in (0, d)])
    ax.set_xlim(-gap / 2, offsets[-1] + durations[-1] + gap / 2)
    ax.set_yticks(np.arange(len(union_ids)) * spacing, [str(u) for u in union_ids])
    ax.set(xlabel="time within session (min)", ylabel="union cell id")
    ax.set_title(f"{reg.name}: {name}", pad=30)
    return fig
