"""A tiny dependency-free PDF writer (Helvetica text, word-wrapped, multi-page).

Used to serve some synthetic corpus documents as real PDF bytes so demo runs and tests
exercise the PDF extraction path end to end.
"""

from __future__ import annotations

import textwrap
import unicodedata

_REPLACEMENTS = {
    "—": "-",
    "–": "-",
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    "×": "x",
    "≥": ">=",
    "≤": "<=",
}
LINES_PER_PAGE = 56
WRAP_COLUMNS = 92


def _ascii(text: str) -> str:
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(text: str, *, title: str = "", author: str = "") -> bytes:
    lines: list[str] = []
    for paragraph in _ascii(text).split("\n"):
        lines.extend(textwrap.wrap(paragraph, WRAP_COLUMNS) or [""])
    pages = [
        lines[i : i + LINES_PER_PAGE] for i in range(0, max(1, len(lines)), LINES_PER_PAGE)
    ] or [[]]

    objects: list[bytes] = []
    n_pages = len(pages)
    font_id = 3 + 2 * n_pages
    info_id = font_id + 1
    page_ids = [3 + 2 * i for i in range(n_pages)]

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    for i, page_lines in enumerate(pages):
        content_id = page_ids[i] + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        ops = ["BT", "/F1 10 Tf", "12 TL", "50 750 Td"]
        for line in page_lines:
            ops.append(f"({_escape(line)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(
        f"<< /Title ({_escape(_ascii(title))}) /Author ({_escape(_ascii(author))}) >>".encode()
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info {info_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
