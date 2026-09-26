"""The paper's evidence set: which run directories its numbers rest on.

arXiv v1 was submitted on 2026-09-22 and every run it reports was logged by
then. Runs logged afterwards (hand-face association tests, new flights) are new
evidence: they belong in outputs/runs/ like any other launch, but they must not
move a published number. Every paper script therefore reads runs through
paper_glob(), never glob() on outputs/runs directly.

Moving the cutoff is a deliberate act, done together with the manuscript
revision that reports the new runs, and audit_claims.py must reproduce every
claim afterwards.
"""
import glob
import os
import re

# Inclusive. Run directories are named <YYYYmmdd-HHMMSS>_<tag>.
PAPER_RUNS_UNTIL = "20260922-235959"

_RUN_DIR = re.compile(r"outputs[\\/]runs[\\/](\d{8}-\d{6})")


def in_paper(path: str) -> bool:
    """True if `path` lies in a run directory logged on or before the cutoff."""
    m = _RUN_DIR.search(path)
    return bool(m) and m.group(1) <= PAPER_RUNS_UNTIL


def paper_glob(pattern: str) -> list:
    """glob.glob(pattern), restricted to runs in the paper's evidence set."""
    return [p for p in glob.glob(pattern) if in_paper(p)]
