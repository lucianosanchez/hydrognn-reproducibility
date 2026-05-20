# Contributing

This repository is the reproducibility package of an EAAI submission.
Day-to-day development happens in a separate private repository; this
public mirror tracks the code, data and result tables that support the
published manuscript.

## Scope of accepted contributions

* **Bug reports** on any of the scripts or modules. Please include the
  full traceback, the command you ran, and your platform.
* **Reproducibility issues** (numbers in `outputs/*.csv` not matching
  the manuscript). Please point to the specific table or paragraph and
  the CSV column.
* **Portability fixes** (e.g. additional Python versions, alternative
  PyTorch builds, conda env tweaks).
* **Documentation clarifications** in `README.md`, `REPRODUCING.md`
  or `docs/`.

We do **not** accept new modelling contributions in this repository.
Methodological extensions of the framework should be developed against
a fork; once they are mature and published they can be linked here.

## How to open an issue or pull request

1. Open an issue first when in doubt. Tag it `bug`, `repro` or
   `docs` so it is easy to triage.
2. For pull requests: fork the repo, branch from `main`, keep changes
   focused on a single concern, and reference the issue.
3. Pre-commit checks expected before opening a PR:
   ```bash
   make smoke                   # imports OK
   bash -n scripts/*.sh         # bash syntax OK
   python -m compileall -q seq2seq_runoff synth_simulator scripts
   ```
4. Numerical CSVs in `outputs/` are the authoritative source for the
   manuscript. Do not modify them without re-running the corresponding
   experiment phase; if you do re-run, include the log
   (`outputs/_logs/*.log`) in the PR.

## Citation discipline

Any reference added in future write-ups must be verified to exist
(DOI, arXiv id or journal page) before being committed. The May 2026
manuscript revision removed eight unverifiable entries and corrected
the metadata of four; the lesson is that AI-assisted drafts should
not be trusted on bibliography without independent verification.

## License

By contributing, you agree that your contribution will be released
under the MIT license that covers the rest of this repository.
