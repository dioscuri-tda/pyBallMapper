"""html_report.py — a tiny, dependency-light builder for a single self-contained
HTML benchmark report.

Matplotlib figures are embedded as base64 PNGs and tables are emitted as
inline-styled HTML, so the produced ``report.html`` is ONE file that opens in
any browser, offline, with no external assets.  Only numpy / matplotlib are
imported (lazily, inside the methods that need them) so this module also loads
on a machine without a GPU or the heavier benchmark dependencies.
"""

from __future__ import annotations

import base64
import html as _html
import io

_CSS = """
:root{
  --fg:#1f2933; --fg-soft:#52606d; --bg:#fafbfc; --panel:#ffffff;
  --border:#e4e7eb; --accent:#2563eb; --accent-soft:#dbeafe;
  --code-bg:#f3f4f6; --good:#047857; --warn:#c2410c; --bad:#b91c1c;
  --stripe:#f9fafb;
}
*{box-sizing:border-box}
body{
  font-family:-apple-system,"Segoe UI","Noto Sans KR",Roboto,Arial,sans-serif;
  color:var(--fg); background:var(--bg); line-height:1.6; font-size:15px;
  margin:0; padding:0;
}
main{max-width:1080px; margin:0 auto; padding:32px 24px 80px}
h1{font-size:25px; margin:0 0 4px}
.subtitle{color:var(--fg-soft); margin:0 0 28px}
h2{
  font-size:20px; margin:44px 0 14px; padding-bottom:6px;
  border-bottom:2px solid var(--accent-soft);
}
p{margin:0 0 14px}
code{background:var(--code-bg); padding:1px 5px; border-radius:4px;
  font-family:"SF Mono",Consolas,monospace; font-size:13px}
table{border-collapse:collapse; width:100%; margin:6px 0 18px; font-size:13.5px}
th,td{border:1px solid var(--border); padding:6px 10px; text-align:right}
th{background:var(--accent-soft); color:var(--fg); font-weight:600}
td:first-child,th:first-child{text-align:left}
tbody tr:nth-child(even){background:var(--stripe)}
.good{color:var(--good); font-weight:600}
.bad{color:var(--bad); font-weight:600}
figure{margin:8px 0 24px}
figure img{max-width:100%; border:1px solid var(--border); border-radius:8px}
figcaption{color:var(--fg-soft); font-size:13px; margin-top:6px}
.callout{
  border-left:4px solid var(--accent); background:var(--accent-soft);
  padding:10px 16px; border-radius:4px; margin:0 0 18px;
}
.callout.warn{border-color:var(--warn); background:#fff7ed}
.callout.good{border-color:var(--good); background:#ecfdf5}
.meta{color:var(--fg-soft); font-size:13px}
"""


class Report:
    """Accumulate report blocks, then ``save()`` a single self-contained file."""

    def __init__(self, title: str, subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self._parts: list[str] = []

    # ── content blocks ──────────────────────────────────────────────────────
    def h2(self, text: str) -> "Report":
        self._parts.append(f"<h2>{_html.escape(text)}</h2>")
        return self

    def p(self, html_text: str) -> "Report":
        self._parts.append(f"<p>{html_text}</p>")
        return self

    def html(self, raw: str) -> "Report":
        self._parts.append(raw)
        return self

    def callout(self, html_text: str, kind: str = "info") -> "Report":
        cls = "callout" if kind == "info" else f"callout {kind}"
        self._parts.append(f'<div class="{cls}">{html_text}</div>')
        return self

    def table(self, headers, rows) -> "Report":
        """``headers``: list[str]; ``rows``: list[list[str]] (pre-formatted,
        may contain inline HTML such as ``<span class='good'>``)."""
        head = "".join(f"<th>{_html.escape(str(h))}</th>" for h in headers)
        body = []
        for r in rows:
            tds = "".join(f"<td>{c}</td>" for c in r)
            body.append(f"<tr>{tds}</tr>")
        self._parts.append(
            "<table><thead><tr>"
            + head
            + "</tr></thead><tbody>"
            + "".join(body)
            + "</tbody></table>"
        )
        return self

    def figure(self, fig, caption: str = "", dpi: int = 130) -> "Report":
        """Embed a matplotlib Figure as a base64 PNG (no external file)."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        cap = f"<figcaption>{caption}</figcaption>" if caption else ""
        self._parts.append(
            f'<figure><img src="data:image/png;base64,{b64}" alt="">{cap}</figure>'
        )
        return self

    def image_file(self, path: str, caption: str = "") -> "Report":
        """Embed an existing PNG file as base64 (used to inline saved figures)."""
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        cap = f"<figcaption>{caption}</figcaption>" if caption else ""
        self._parts.append(
            f'<figure><img src="data:image/png;base64,{b64}" alt="">{cap}</figure>'
        )
        return self

    # ── output ──────────────────────────────────────────────────────────────
    def save(self, path: str) -> str:
        sub = f'<p class="subtitle">{self.subtitle}</p>' if self.subtitle else ""
        doc = (
            "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{_html.escape(self.title)}</title>\n"
            f"<style>{_CSS}</style>\n</head>\n<body>\n<main>\n"
            f"<h1>{_html.escape(self.title)}</h1>\n{sub}\n"
            + "\n".join(self._parts)
            + "\n</main>\n</body>\n</html>\n"
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(doc)
        return path
