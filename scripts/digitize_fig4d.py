"""Digitize the human data (circles) of Weiss et al. (2002) Fig. 4d from the PDF's vector paths.

Each circle is drawn as four Bezier arcs; its centre is the mean of the four arc end points.
Pixels are mapped to data with the axis tick positions read from the same page (x: 0-40 deg rhombus
angle, y: 0-50 deg direction, y axis pointing down). Writes data/digitized/weiss_fig4d.csv and
figures/check_digitized_fig4d.png (extracted points in red on the rendered figure).
"""
import numpy as np
import pymupdf
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = pymupdf.Rect(225, 355, 360, 470)        # Fig. 4d on page 601 (PDF page index 3), in points


def tick_map(page):
    """Linear maps from page points to data units, using the numeric tick labels of the panel."""
    words = [w for w in page.get_text('words') if PANEL.contains(pymupdf.Rect(w[:4])) and w[4].isdigit()]
    xt = [((w[0] + w[2]) / 2, float(w[4])) for w in words if w[1] > 454]          # labels under the x axis
    yt = [((w[1] + w[3]) / 2, float(w[4])) for w in words if w[2] < 237.3 and w[1] < 454 or (w[4] == '50' and w[0] < 235)]
    fx = np.polyfit([p for p, _ in xt], [v for _, v in xt], 1)
    fy = np.polyfit([p for p, _ in yt], [v for _, v in yt], 1)
    return fx, fy, xt, yt


def circle_centres(page):
    """Centres (page points) of every closed 4-arc path inside the panel."""
    centres = []
    for d in page.get_drawings():
        if not PANEL.contains(d['rect']):
            continue
        arcs = [it for it in d['items'] if it[0] == 'c']
        for i in range(0, len(arcs) - 3, 4):
            pts = np.array([[it[4].x, it[4].y] for it in arcs[i:i + 4]])
            centres.append(pts.mean(axis=0))
    return np.array(centres)


def main():
    page = pymupdf.open(ROOT / 'papers/weiss2002.pdf')[3]
    fx, fy, xt, yt = tick_map(page)
    c = circle_centres(page)
    angle, direction = np.polyval(fx, c[:, 0]), np.polyval(fy, c[:, 1])
    order = np.argsort(angle)
    out = ROOT / 'data/digitized/weiss_fig4d.csv'
    np.savetxt(out, np.column_stack([angle[order], direction[order]]), delimiter=',', fmt='%.3f',
               header='rhombus_angle_deg,perceived_direction_deg', comments='')

    zoom = 8
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=PANEL)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img)
    ax.plot((c[:, 0] - PANEL.x0) * zoom, (c[:, 1] - PANEL.y0) * zoom, 'o', mfc='none', mec='red', ms=9, mew=1.5)
    for a, dval, (px, py) in zip(angle, direction, c):
        ax.annotate(f'({a:.1f}, {dval:.1f})', ((px - PANEL.x0) * zoom, (py - PANEL.y0) * zoom),
                    xytext=(8, -8), textcoords='offset points', color='red', fontsize=8)
    ax.set_title('Weiss et al. 2002 Fig. 4d: extracted circles (red) on the original', fontsize=10)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(ROOT / 'figures/check_digitized_fig4d.png', dpi=200)
    print(f'{len(c)} circles; x ticks {xt}; y ticks {yt}')
    print(np.column_stack([angle[order], direction[order]]).round(2))


if __name__ == '__main__':
    main()
