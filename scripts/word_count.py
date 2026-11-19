"""Count report words, excluding references and the text inside figure images.

Reads report/main.tex from the abstract to the end, replaces each number macro with its value from
numbers.tex, drops the bibliography commands and LaTeX markup, and counts the remaining words
(captions and tables included)."""
import re
from eye2echo.style import ROOT, RES


def main():
    nums = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{(.*)\}', (ROOT / 'report/numbers.tex').read_text()))
    tex = (ROOT / 'report/main.tex').read_text()
    body = tex[tex.index('\\begin{abstract}'):]
    body = re.sub(r'(?<!\\)%.*', '', body)
    body = re.sub(r'\\(begin|end|includegraphics|label|ref|cite|bibliographystyle|bibliography|graphicspath)'
                  r'(\[[^]]*\])?\{[^}]*\}', ' ', body)
    body = re.sub(r'\{(l|c)+\}', ' ', body)                                     # table column specs
    body = re.sub(r'\\(\w+)', lambda m: ' ' + re.sub(r'\\ensuremath', '', nums.get(m.group(1), '')) + ' ', body)
    words = len(re.findall(r'[A-Za-z0-9][A-Za-z0-9.\-]*', re.sub(r'[{}$^_\\]', ' ', body)))
    (RES / 'word_count.txt').write_text(f'{words}\n')
    print(words)


if __name__ == '__main__':
    main()
