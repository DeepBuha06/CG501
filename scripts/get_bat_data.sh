#!/usr/bin/env bash
# Fetch the public data and MATLAB code of Salles, Diebold & Moss (2020, PNAS).
# Source: https://github.com/angiesalles/batTargetPrediction (no licence file: cite it, do not redistribute).
# The head-tracking CSVs and trial sheets are plain files. The .mat files are Git LFS objects;
# we try to fetch only the small call-time files (data/shouts*.mat, under 200 KB each).
set -e
mkdir -p data/external
cd data/external
if [ ! -d battargetprediction ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/angiesalles/batTargetPrediction battargetprediction
fi
cd battargetprediction
if command -v git-lfs >/dev/null 2>&1; then
  git lfs install --local
  git lfs pull --include="data/shouts.mat,data/shoutsFast.mat,data/shoutsControls.mat" || echo "LFS pull failed: call times unavailable, use the approximate-call-time path."
else
  echo "git-lfs not installed: run 'sudo apt install git-lfs' and re-run this script to get the call times."
fi
for f in data/shouts.mat data/shoutsFast.mat data/shoutsControls.mat; do
  if head -c 40 "$f" | grep -q "git-lfs"; then echo "MISSING (LFS pointer only): $f"; else echo "OK: $f ($(stat -c %s "$f") bytes)"; fi
done
echo "Head-tracking CSVs: $(find data -name '*_xyzpts.csv' | wc -l)"
