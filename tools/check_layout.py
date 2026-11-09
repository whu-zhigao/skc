#!/usr/bin/env python3
"""Report where the main text ends relative to the 12-page limit.

Prints total pages, the page on which REFERENCES starts, and how far down
that page the REFERENCES heading sits (rough proxy for main-text fill).
"""
from __future__ import annotations

import sys

PDF = "main_icde.pdf"

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except ImportError:
        sys.exit("no pypdf/PyPDF2 available")

reader = PdfReader(PDF)
n = len(reader.pages)
print(f"total pages: {n}")

for probe in (11, 12, 13):
    if probe <= n:
        t = reader.pages[probe - 1].extract_text() or ""
        print(f"--- page {probe} head: {t[:160]!r}")

# The IEEEtran heading is uppercase; body occurrences are lowercase.
last = None
for i, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    pos = text.find("REFERENCES")
    if pos != -1:
        last = (i, pos, text)
if last:
    i, pos, text = last
    print(f"REFERENCES heading on page {i}, char {pos} of {len(text)} "
          f"(~{pos / max(len(text),1):.0%} into extracted text)")
    print("...main text ends with:",
          repr(text[max(0, pos - 200):pos].strip()[-140:]))
else:
    print("REFERENCES heading not found")
