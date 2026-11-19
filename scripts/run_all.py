"""Regenerate every result, figure, number and video from scratch.

Usage: python scripts/run_all.py [--clean]   (--clean first deletes results/, figures/, videos/ and the
trial cache, so everything, including report/report_paper_style.pdf, is rebuilt from the papers and the authors' raw data.)
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ['digitize_fig4d.py', 'human_experiments.py', 'real_data.py', 'bat_simulation.py',
         'collect_numbers.py', 'make_videos.py', 'word_count.py']

if __name__ == '__main__':
    if '--clean' in sys.argv:
        for d in ('results', 'figures', 'videos'):
            shutil.rmtree(ROOT / d, ignore_errors=True)
        (ROOT / 'data/external/trials_cache.pkl').unlink(missing_ok=True)
    for d in ('results', 'figures', 'videos'):
        (ROOT / d).mkdir(exist_ok=True)
    for step in STEPS:
        if step == 'digitize_fig4d.py' and not (ROOT / 'papers/weiss2002.pdf').exists():
            print('== digitize_fig4d.py skipped (papers/weiss2002.pdf missing; using data/digitized/weiss_fig4d.csv)')
            continue
        print(f'== {step}', flush=True)
        subprocess.run([sys.executable, '-W', 'ignore', str(ROOT / 'scripts' / step)], check=True,
                       stdout=open(ROOT / 'results' / f'log_{Path(step).stem}.txt', 'w'),
                       env={**os.environ, 'MPLBACKEND': 'Agg'})   # figures are saved, never shown
        if step == 'collect_numbers.py':   # build the report as soon as numbers.tex exists
            subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-silent', 'paper.tex'],
                           cwd=ROOT / 'report', check=True, stdout=subprocess.DEVNULL)
            shutil.copy(ROOT / 'report/paper.pdf', ROOT / 'report/report_paper_style.pdf')
    print('all steps finished')
