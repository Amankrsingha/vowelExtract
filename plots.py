# vowelpipe/plots.py
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')                      # no display needed on a server
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from d3blocks import D3Blocks
from .normalise import ellipse_params

def interactive_vowel_space(df, xcol, ycol, filepath):
    '''Interactive scatter (HTML). df must contain vowel, speaker, word, f1, f2
    (raw Hz, no missing values) and the coordinate columns xcol (F2) and ycol (F1).'''
    tooltip = (df['vowel'] + '<br />' + df['speaker'] + ' / ' + df['word'].fillna('')
               + '<br />F1 ' + df['f1'].round(0).astype(int).astype(str)
               + ' Hz, F2 ' + df['f2'].round(0).astype(int).astype(str) + ' Hz')
    d3 = D3Blocks()
    d3.scatter((-df[xcol]).to_numpy(), (-df[ycol]).to_numpy(),
               size=np.full(len(df), 8),
               color=df['vowel'].to_numpy(),
               tooltip=tooltip.to_numpy(),
               opacity=0.6, stroke='#000000', cmap='tab20',
               filepath=str(filepath))

def static_vowel_space(df, xcol, ycol, path, coverage=0.95,
                       xlabel='F2', ylabel='F1'):
    '''Publication figure with tokens, category ellipses and labels.'''
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colours = plt.cm.tab20.colors
    for i, (vowel, g) in enumerate(df.groupby('vowel')):
        c = colours[i % len(colours)]
        ax.scatter(g[xcol], g[ycol], s=8, alpha=0.35, color=c)
        e = ellipse_params(g[xcol], g[ycol], coverage)
        if e:
            ax.add_patch(Ellipse((e['cx'], e['cy']), e['width'], e['height'],
                                 angle=e['angle'], fill=False, edgecolor=c, linewidth=1.5))
            ax.text(e['cx'], e['cy'], vowel, ha='center', va='center', fontweight='bold')
    ax.invert_xaxis()                      # F2 increases leftwards
    ax.invert_yaxis()                      # F1 increases downwards
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)
