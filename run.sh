#!/usr/bin/env bash
# Short commands for everything in this project. Run from this folder:  bash run.sh <command>
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
if [ ! -x "$PY" ]; then echo "First run: setting up (once)..."; bash setup.sh; fi

case "$1" in
  human)  $PY scripts/live_human.py ;;
  bat)    $PY scripts/live_bat.py ;;
  test)   $PY -m pytest -q ;;
  videos) MPLBACKEND=Agg $PY scripts/make_videos.py && echo "Videos are in videos/" ;;
  report) cd report && latexmk -pdf -interaction=nonstopmode -silent paper.tex && cp paper.pdf report_paper_style.pdf && cd .. \
            && echo "report/report_paper_style.pdf rebuilt. Words (limit 3000):" && $PY scripts/word_count.py ;;
  all)    $PY scripts/run_all.py --clean ;;
  *)
    echo "Usage: bash run.sh <command>"
    echo "  human   live human simulation (rhombus; brightness slider, Run button)"
    echo "  bat     live bat simulation (wall, slow-down, echo noise, Run button)"
    echo "  test    the 10 known-answer tests (about 10 s)"
    echo "  videos  re-make the three MP4 videos (about 4 min)"
    echo "  report  rebuild the report PDF and print the word count"
    echo "  all     rebuild every result, figure, video and the report from scratch (about 6 min)"
    ;;
esac
