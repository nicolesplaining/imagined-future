"""Apply small layout fixes to the Pandoc Word export."""

from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


TABLE_ONE_CAPTION = "Method and controls on one scale."


def keep_row_together(row) -> None:
    properties = row._tr.get_or_add_trPr()
    if properties.find(qn("w:cantSplit")) is None:
        properties.append(OxmlElement("w:cantSplit"))


def repeat_header(row) -> None:
    properties = row._tr.get_or_add_trPr()
    if properties.find(qn("w:tblHeader")) is None:
        header = OxmlElement("w:tblHeader")
        header.set(qn("w:val"), "true")
        properties.append(header)


def main(path: Path) -> None:
    document = Document(path)
    caption = next(
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.startswith(TABLE_ONE_CAPTION)
    )
    caption.paragraph_format.page_break_before = True
    caption.paragraph_format.keep_with_next = True

    primary_table = document.tables[0]
    primary_table.autofit = False
    column_widths = (1.10, 0.85, 0.75, 1.35, 2.45)
    for row in primary_table.rows:
        for cell, width in zip(row.cells, column_widths):
            cell.width = Inches(width)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1
                for run in paragraph.runs:
                    run.font.size = Pt(8)
    repeat_header(primary_table.rows[0])
    for row in primary_table.rows:
        keep_row_together(row)

    document.save(path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: postprocess_docx.py PATH")
    main(Path(sys.argv[1]))
