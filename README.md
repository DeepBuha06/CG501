# From Eye to Echo

CG-501 Computation and Cognition, IIT Gandhinagar. End-term project by Deep Buha (24110082).

Weiss, Simoncelli & Adelson (2002) explain human motion illusions with one Bayesian estimator: measurements
are noisy, and slow motion is more likely than fast. I rebuilt that estimator, wrote it for echoes instead of
images (it becomes a line fit with a slow-motion prior), and tested it in a simulation of the Salles, Diebold
& Moss (2020) bat experiment and on their public head-tracking data.

Report: [`report/report.pdf`](report/report.pdf) (two-column version: [`report/report_paper_style.pdf`](report/report_paper_style.pdf)).

## Running it

Windows (PowerShell, needs WSL with Ubuntu) or Linux/WSL (bash). The first command sets everything up.

| | PowerShell | bash |
|---|---|---|
| Setup | `.\setup.bat` | `bash setup.sh` |
| Live human demo | `.\run.bat human` | `bash run.sh human` |
| Live bat demo | `.\run.bat bat` | `bash run.sh bat` |
| Tests | `.\run.bat test` | `bash run.sh test` |
| Videos | `.\run.bat videos` | `bash run.sh videos` |
| Report PDFs | `.\run.bat report` | `bash run.sh report` |
| Everything from scratch (3 to 6 min) | `.\run.bat all` | `bash run.sh all` |

Setup installs missing system tools (Python, ffmpeg, LaTeX, git-lfs), the pinned packages from
`requirements-lock.txt` into `.venv/`, and downloads the bat data into `data/external/`.

## Layout

```
src/eye2echo/   weiss.py (Weiss Eq. 1), stimuli.py, bat_models.py, bat_sim.py, bat_data.py, stats.py, anim/
scripts/        one script per analysis; run_all.py runs them in order
tests/          known-answer tests
data/digitized/ six human data points read from Weiss et al. Fig. 4d
results/        numbers.json and CSVs written by the scripts
figures/        figures (PDF and PNG)
videos/         the three simulation videos
report/         main.tex (one column), paper.tex (two columns), refs.bib, numbers.tex (generated)
```

## Papers and data

- Weiss, Simoncelli & Adelson (2002). Motion illusions as optimal percepts. *Nature Neuroscience* 5, 598-604. https://doi.org/10.1038/nn858
- Salles, Diebold & Moss (2020). Echolocating bats accumulate information from acoustic snapshots to predict auditory object motion. *PNAS* 117, 29229-29238. https://doi.org/10.1073/pnas.2011719117
- Salles, Diebold & Moss (2021). Bat target tracking strategies for prey interception. *Communicative & Integrative Biology* 14, 37-40. https://doi.org/10.1080/19420889.2021.1898751
- Stocker & Simoncelli (2005). Constraining a Bayesian model of human visual speed perception. *NeurIPS 17*, 1361-1368.

The PDFs are not included because of copyright. To re-read the Fig. 4d points, save the Weiss paper as
`papers/weiss2002.pdf` and run `bash run.sh all`.

Bat data and MATLAB code: https://github.com/angiesalles/batTargetPrediction. That repository has no
licence, so its data are downloaded by `scripts/get_bat_data.sh` and not copied here. `bat_data.py` follows
the trial selection in their `miroTracking.m`.
