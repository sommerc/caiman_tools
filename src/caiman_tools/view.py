"""Render an interactive bokeh component browser for CaImAn estimates.

Adapted from caiman.utils.visualization.nb_view_patches (the engine behind
Estimates.nb_view_components used interactively in demo_pipeline_cnmfE.ipynb),
targeting a standalone HTML file instead of a Jupyter notebook, and extended
to additionally plot the deconvolved (S) and detrended (F_dff) traces
alongside the raw and denoised ones, each with its own color and legend label.
"""

from __future__ import annotations

import contextlib
import webbrowser
from pathlib import Path
from typing import Any

import bokeh
import bokeh.layouts
import bokeh.plotting as bpl
import matplotlib
import matplotlib.colors
import numpy as np
import scipy.sparse
from bokeh.models import ColumnDataSource, CustomJS, LabelSet, Range1d, Slider
from loguru import logger

from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.utils.visualization import get_contours

_WHICH_TO_IDX_ATTR = {
    "accepted": "idx_components",
    "rejected": "idx_components_bad",
}

_TRACE_STYLE = {
    "raw": dict(color="#7f7f7f", alpha=0.6, label="raw (C+YrA)"),
    "denoised": dict(color="#d62728", alpha=0.9, label="denoised (C)"),
    "deconvolved": dict(color="#2ca02c", alpha=0.9, label="deconvolved (S)"),
    "detrended": dict(color="#1f77b4", alpha=0.9, label="detrended (F_dff)"),
}


@contextlib.contextmanager
def _suppress_browser_autoopen():
    """bokeh's show() tries to auto-launch a system browser, which can hang
    indefinitely on headless/remote setups (observed on this cluster over
    X11 forwarding). Neutralize it for the duration of the call; the CLI
    layer does its own explicit, detached open if the user opts in."""
    originals = (webbrowser.open, webbrowser.open_new, webbrowser.open_new_tab)
    webbrowser.open = lambda *a, **k: True
    webbrowser.open_new = lambda *a, **k: True
    webbrowser.open_new_tab = lambda *a, **k: True
    try:
        yield
    finally:
        webbrowser.open, webbrowser.open_new, webbrowser.open_new_tab = originals


def _select(arr: Any, idx: np.ndarray | None) -> Any:
    if arr is None:
        return None
    return arr[idx] if idx is not None else arr


def _render_component_viewer(
    A,
    C: np.ndarray,
    S: np.ndarray | None,
    F_dff: np.ndarray | None,
    YrA: np.ndarray | None,
    d1: int,
    d2: int,
    image_neurons: np.ndarray | None = None,
    thr: float = 0.9,
    cmap: str = "viridis",
    r_values: np.ndarray | None = None,
    SNR: np.ndarray | None = None,
    cnn_preds: np.ndarray | None = None,
) -> None:
    if not scipy.sparse.issparse(A):
        A = scipy.sparse.csc_matrix(A)

    nr, T = C.shape
    raw = C + YrA if YrA is not None else C.copy()
    deconvolved = S if S is not None else np.zeros_like(C)
    detrended = F_dff if F_dff is not None else np.zeros_like(C)

    x = np.arange(T)
    if image_neurons is None:
        image_neurons = np.array(A.mean(axis=1)).reshape((d1, d2), order="F")

    coors = get_contours(A, (d1, d2), thr)
    cc1 = [cor["coordinates"][:, 0] for cor in coors]
    cc2 = [cor["coordinates"][:, 1] for cor in coors]
    c1, c2 = cc1[0], cc2[0]

    scale = 100.0
    source = ColumnDataSource(
        data=dict(
            x=x,
            y_raw=raw[0] / scale,
            y_denoised=C[0] / scale,
            y_deconv=deconvolved[0] / scale,
            y_detrend=detrended[0] / scale,
        )
    )
    source_ = ColumnDataSource(
        data=dict(
            z_raw=raw / scale,
            z_denoised=C / scale,
            z_deconv=deconvolved / scale,
            z_detrend=detrended / scale,
        )
    )
    source2 = ColumnDataSource(data=dict(c1=c1, c2=c2))
    source2_ = ColumnDataSource(data=dict(cc1=cc1, cc2=cc2))

    code = """
        var data = source.data
        var data_ = source_.data
        var f = cb_obj.value - 1
        var x = data['x']

        for (var i = 0; i < x.length; i++) {
            data['y_raw'][i] = data_['z_raw'][i + f * x.length]
            data['y_denoised'][i] = data_['z_denoised'][i + f * x.length]
            data['y_deconv'][i] = data_['z_deconv'][i + f * x.length]
            data['y_detrend'][i] = data_['z_detrend'][i + f * x.length]
        }

        var data2_ = source2_.data
        var data2 = source2.data
        var c1 = data2['c1']
        var c2 = data2['c2']
        var cc1 = data2_['cc1']
        var cc2 = data2_['cc2']

        for (var i = 0; i < c1.length; i++) {
            c1[i] = cc1[f][i]
            c2[i] = cc2[f][i]
        }
        source2.change.emit()
        source.change.emit()
    """

    if r_values is not None:
        code += """
            var mets = metrics.data['mets']
            mets[1] = metrics_.data['R'][f].toFixed(3)
            mets[2] = metrics_.data['SNR'][f].toFixed(3)
            metrics.change.emit();
        """
        metrics = ColumnDataSource(
            data=dict(
                y=(3, 2, 1, 0),
                mets=(
                    "",
                    "% 7.3f" % r_values[0],
                    "% 7.3f" % SNR[0],
                    "N/A" if np.sum(cnn_preds) in (0, None) else "% 7.3f" % cnn_preds[0],
                ),
                keys=("Evaluation Metrics", "Spatial corr:", "SNR:", "CNN:"),
            )
        )
        if np.sum(cnn_preds) in (0, None):
            metrics_ = ColumnDataSource(data=dict(R=r_values, SNR=SNR))
        else:
            metrics_ = ColumnDataSource(data=dict(R=r_values, SNR=SNR, CNN=cnn_preds))
            code += "mets[3] = metrics_.data['CNN'][f].toFixed(3)\n"
        labels = LabelSet(x=0, y="y", text="keys", source=metrics)
        labels2 = LabelSet(x=10, y="y", text="mets", source=metrics, text_align="right")
        plot2 = bpl.figure(width=200, height=100, toolbar_location=None)
        plot2.axis.visible = False
        plot2.grid.visible = False
        plot2.tools.visible = False
        plot2.line([0, 10], [0, 4], line_alpha=0)
        plot2.add_layout(labels)
        plot2.add_layout(labels2)
    else:
        metrics, metrics_ = None, None

    callback = CustomJS(
        args=dict(
            source=source, source_=source_, source2=source2, source2_=source2_,
            metrics=metrics, metrics_=metrics_,
        ),
        code=code,
    )

    plot = bpl.figure(width=600, height=300)
    for key, ycol in (
        ("raw", "y_raw"),
        ("denoised", "y_denoised"),
        ("deconvolved", "y_deconv"),
        ("detrended", "y_detrend"),
    ):
        style = _TRACE_STYLE[key]
        plot.line(
            "x", ycol, source=source, line_width=1, line_alpha=style["alpha"],
            color=style["color"], legend_label=style["label"],
        )
    plot.legend.click_policy = "hide"
    plot.legend.label_text_font_size = "8pt"
    plot.legend.background_fill_alpha = 0.6

    colormap = matplotlib.colormaps.get_cmap(cmap)
    grayp = [matplotlib.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]

    xr = Range1d(start=0, end=image_neurons.shape[1])
    yr = Range1d(start=image_neurons.shape[0], end=0)
    plot1 = bpl.figure(
        x_range=xr, y_range=yr,
        width=int(min(1, d2 / d1) * 300), height=int(min(1, d1 / d2) * 300),
    )
    plot1.image(image=[image_neurons], x=0, y=0, dw=d2, dh=d1, palette=grayp)
    plot1.patch("c1", "c2", alpha=0.6, color="purple", line_width=2, source=source2)

    if nr > 1:
        slider = Slider(start=1, end=nr, value=1, step=1, title="Neuron Number")
        slider.js_on_change("value", callback)
        bpl.show(
            bokeh.layouts.layout(
                [[slider], [bokeh.layouts.row(
                    plot1 if metrics is None else bokeh.layouts.column(plot1, plot2), plot
                )]]
            )
        )
    else:
        bpl.show(
            bokeh.layouts.row(plot1 if metrics is None else bokeh.layouts.column(plot1, plot2), plot)
        )


def render_estimates_view(
    estimates,
    which: str = "accepted",
    cmap: str = "viridis",
    thr: float = 0.9,
    output_html: Path | None = None,
    title: str = "components",
) -> Path:
    """Write an interactive bokeh viewer for an already-loaded Estimates object."""
    if which == "all":
        idx = None
        n = len(estimates.C)
    else:
        idx = getattr(estimates, _WHICH_TO_IDX_ATTR[which])
        n = len(idx)

    if n == 0:
        raise ValueError(f"No '{which}' components -- nothing to display.")

    A = estimates.A.tocsc()[:, idx] if idx is not None else scipy.sparse.csc_matrix(estimates.A)

    # cnn_preds is `[]` (not per-component) when use_cnn=False; treat as absent.
    cnn_preds = estimates.cnn_preds
    if cnn_preds is None or len(cnn_preds) != len(estimates.C):
        cnn_preds = None
    else:
        cnn_preds = _select(cnn_preds, idx)

    bpl.output_file(str(output_html), title=title)
    with _suppress_browser_autoopen():
        _render_component_viewer(
            A=A,
            C=_select(estimates.C, idx),
            S=_select(estimates.S, idx),
            F_dff=_select(estimates.F_dff, idx),
            YrA=_select(estimates.YrA, idx),
            d1=estimates.dims[0],
            d2=estimates.dims[1],
            image_neurons=estimates.Cn,
            thr=thr,
            cmap=cmap,
            r_values=_select(estimates.r_values, idx),
            SNR=_select(estimates.SNR_comp, idx),
            cnn_preds=cnn_preds,
        )

    logger.info("Wrote interactive component viewer to {} ({} '{}' components)", output_html, n, which)
    return output_html


def view_components(
    hdf5_path: Path,
    which: str = "accepted",
    cmap: str = "viridis",
    thr: float = 0.9,
    output_html: Path | None = None,
) -> Path:
    """Load a CaImAn hdf5 result and write its interactive component viewer."""
    model = load_CNMF(str(hdf5_path))
    if output_html is None:
        output_html = hdf5_path.parent / f"{hdf5_path.stem}_view.html"
    return render_estimates_view(
        model.estimates, which=which, cmap=cmap, thr=thr,
        output_html=output_html, title=f"{hdf5_path.name} ({which})",
    )
