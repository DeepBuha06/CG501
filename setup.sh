#!/usr/bin/env bash
# One-time setup on Ubuntu or WSL. Run it from this folder:  bash setup.sh
# Installs the system tools (only if missing), the Python packages in .venv, and the authors' bat data.
set -e
cd "$(dirname "$0")"

echo "== 1/4 System tools"
if command -v ffmpeg >/dev/null && command -v latexmk >/dev/null && command -v git-lfs >/dev/null \
   && python3 -c "import tkinter, ensurepip" 2>/dev/null; then
  echo "already installed"
else
  echo "Installing (Ubuntu will ask for your password once)..."
  sudo apt-get update -qq
  sudo apt-get install -y python3-venv python3-tk ffmpeg git git-lfs latexmk \
       texlive-latex-extra texlive-fonts-recommended
fi

echo "== 2/4 Python packages (into .venv, a few minutes the first time)"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements-lock.txt || .venv/bin/pip install -q -r requirements.txt
.venv/bin/pip install -q -e .

echo "== 3/4 Bat data (Salles et al. 2020, about 350 MB the first time)"
bash scripts/get_bat_data.sh

echo "== 4/4 Check"
.venv/bin/python -m pytest -q

echo
echo "Setup finished. Next: bash run.sh human    (or: bash run.sh  to see every command)"
