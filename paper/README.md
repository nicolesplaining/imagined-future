# NeurIPS 2026 LaTeX source

This folder contains the concise paper revision, the NeurIPS 2026 style file supplied with the draft, the bibliography, the conference checklist, and the inputs used to generate the figures. `REVISION_NOTES.md` explains the structural changes and remaining submission blockers.

## Submission mode

The source currently uses the official `dblblindworkshop` option and identifies Trustworthy AI for Good as the target workshop. The compiled submission PDF is anonymous.

- For a double-blind workshop, use `dblblindworkshop` and add `\workshoptitle{WORKSHOP NAME}` after loading the style.
- For a single-blind workshop, use `sglblindworkshop` and add the workshop title.
- For a public author-visible preprint, use `preprint`.

Use the target workshop's call for papers to determine its page limit, anonymity policy, and checklist requirements.

## Build

```bash
python make_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The figures are generated from `results/paper_summary.json`. Values without reported confidence intervals are stored as means only and are drawn without error bars.

## Files

- `main.tex`: complete paper and appendices
- `REVISION_NOTES.md`: corpus findings, revision decisions, and unresolved issues
- `references.bib`: bibliography checked against primary sources
- `neurips_2026.sty`: supplied conference style
- `checklist.tex`: NeurIPS checklist
- `make_figures.py`: figure generator
- `results/paper_summary.json`: numeric inputs used by the figures

The project compiles with Python 3, Matplotlib, and a standard TeX Live installation with `latexmk`.
