"""
Save a list of Matplotlib Figures either to a multi-page PDF
or a single self-contained HTML file (Plotly).

Public API
----------
save(figs: list[Figure], out_path: str)
"""

from __future__ import annotations
from typing import List, Union
import matplotlib.pyplot as plt
import base64
import io
from pathlib import Path


# -----------------------------------------------------------------------------
def _fig_to_png_base64(fig) -> str:
    """Return a base64-encoded PNG string from a Matplotlib Figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _to_html(fig: plt.Figure) -> str:
    """Convert a Matplotlib fig to Plotly HTML snippet (no JS duplication)."""
    import plotly.io as pio
    from plotly.tools import mpl_to_plotly  # type: ignore

    plo = mpl_to_plotly(fig)
    return pio.to_html(plo, full_html=False, include_plotlyjs="cdn")


# -----------------------------------------------------------------------------


def save(
    figs: List[plt.Figure], output_dir: Union[str, Path], report_format: str = "pdf"
) -> None:
    """
    Save individual figures under <output_dir>/artifacts and build a combined
    PDF or HTML report at <output_dir>/full_report.{report_format}.
    """
    report_format = report_format.lower()
    if report_format not in {"pdf", "html"}:
        raise ValueError("report_format must be 'pdf' or 'html'")

    output_dir = Path(output_dir)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Save all figures individually (PNG is agnostic to final report type) ──
    for i, fig in enumerate(figs, 1):
        fig_path = artifacts_dir / f"fig_{i:03d}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved fig {i:03d} to {fig_path}")

    # ── 2. Assemble the full report ──
    report_path = output_dir / f"full_report.{report_format}"

    if report_format == "pdf":
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(report_path) as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches="tight")
        print(f"[reporting] PDF saved → {report_path}")

    elif report_format == "html":
        img_tags = []
        for img_file in sorted(artifacts_dir.glob("fig_*.png")):
            b64 = base64.b64encode(img_file.read_bytes()).decode()
            img_tags.append(
                f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:100%; height:auto;">'
            )
        html_doc = (
            "<html><body style='font-family:Helvetica,Arial,sans-serif;'>\n"
            + "<hr/>\n".join(img_tags)
            + "\n</body></html>"
        )
        report_path.write_text(html_doc, encoding="utf-8")
        print(f"[reporting] HTML saved → {report_path}")
