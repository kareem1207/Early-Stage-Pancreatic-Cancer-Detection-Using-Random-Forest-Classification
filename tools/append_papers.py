"""Append the full text of the reference papers to the end of the report PDF.

Download the two papers yourself (links are in Appendix A of the report), then:

    uv run --with pypdf python tools/append_papers.py \
        reports/research_paper.pdf debernardi2020.pdf breiman2001.pdf \
        -o reports/research_paper_with_papers.pdf

Each paper's pages are added after the report, with a PDF bookmark per paper.
"""
import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", help="the report PDF (e.g. reports/research_paper.pdf)")
    parser.add_argument("papers", nargs="+", help="paper PDFs to append, in order")
    parser.add_argument("-o", "--output", default="research_paper_with_papers.pdf")
    args = parser.parse_args()

    writer = PdfWriter()
    for pdf in [args.report, *args.papers]:
        if not Path(pdf).is_file():
            raise SystemExit(f"File not found: {pdf}")

    writer.append(PdfReader(args.report))
    for pdf in args.papers:
        start = len(writer.pages)
        writer.append(PdfReader(pdf))
        writer.add_outline_item(f"Reference paper: {Path(pdf).stem}", start)

    with open(args.output, "wb") as f:
        writer.write(f)
    print(f"Wrote {args.output} ({len(writer.pages)} pages)")


if __name__ == "__main__":
    main()
