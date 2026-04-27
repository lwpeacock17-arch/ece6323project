# ECE6323 Project 2 Solver

This workspace now includes a runnable solver for the "Intelligent Merging Unit" current-instrumentation-channel project. It reads COMTRADE data, solves the 15-state quadratized dynamic state estimation model with a Gauss-Newton loop, performs the chi-square confidence calculation, and writes a small report bundle.

## Files

- `solver.py`: main CLI, COMTRADE parser, estimator, chi-square test, and report generation.
- `make_dashboard.py`: creates the tabbed HTML dashboard from the generated event outputs.
- `../../report/pythonVersion_report.tex`: Overleaf-ready report draft for the Python implementation.

## Quick Start

Synthetic smoke test:

```bash
python3 solver.py --synthetic --limit 80
```

Real COMTRADE event:

```bash
python3 solver.py /path/to/ECE6323_PROJECT_02_CTCHANNEL_EVENT_01.cfg
```

Or, if the `.dat` file is not beside the `.cfg` file:

```bash
python3 solver.py /path/to/event.cfg /path/to/event.dat
```

Useful options:

- `--channel 1`: choose the analog channel used as `vout`.
- `--limit 500`: only process the first N samples.
- `--output-dir outputs/event01`: override the report directory.
- `--max-iter 40`: allow more Gauss-Newton iterations per sample.

## Outputs

Each run creates:

- `results.csv`: per-sample estimates and confidence values.
- `residuals.csv`: per-sample measurement residuals for all 21 equations.
- `summary.json`: compact run summary.
- `report.html`: lightweight visual report with inline SVG plots.

The repository branch also includes:

- `report/outputs/event01/` and `report/outputs/event02/`: CSV/JSON outputs used by the LaTeX report.
- `report/outputs/dashboard.html`: tabbed event dashboard for visual review.

## Notes

- The solver currently supports ASCII COMTRADE `.dat` files.
- The measurement model follows the quadratized 21-measurement / 15-state formulation in Appendix A of the assignment.
- Time derivatives are approximated using the previous solved state and the current sample interval.
