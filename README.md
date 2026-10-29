# Project C19 -- Blind Image Deblurring with a Compact Transformer

Course project (Digital Image Processing & Video Processing).
Team: Zainab Ahmed, Rawda Abdelkhalek.

## Goal

Remove motion blur from a single image with a compact (parameter-budgeted) transformer or
hybrid CNN-transformer, and compare accuracy/efficiency against a classical Wiener-filter
baseline.

## Status

| Milestone                             | Status      |
| ------------------------------------- | ----------- |
| M1 proposal + Wiener baseline numbers | in progress |
| M2 progress report                    | not started |
| M3 final report                       | not started |

No results have been measured yet. Numbers will be added here only after they are produced by the code in this repo.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ is required (course standard stack).

## Repository layout

```
src/          source code (utils.py holds the global random seed)
configs/      experiment settings
data/         dataset instructions + subset records (data itself is git-ignored)
notebooks/    exploration notebooks
results/      figures/ and tables/ produced by experiments
report/       LaTeX sources and references.bib
```

## Reproducibility

- Fixed seed: defined once in `src/utils.py`.
- Exact dataset subsets (count, selection method, seed): recorded in `data/README.md`.
- Dependencies pinned in `requirements.txt`.

## AI-assistance disclosure

Code scaffolding was AI-assisted (Claude). All codes, reported results must be run and verified by the team.

## Contributions

Each member commits from her own account. See the work-breakdown in the report.
