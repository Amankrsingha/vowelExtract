# vowelpipe/validate.py
# 1) python -c "from vowelpipe.validate import make_sample_sheet; make_sample_sheet()"
# 2) fill in output/manual_sample.csv by hand in Praat
# 3) python -m vowelpipe.validate output/manual_sample.csv
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import config as cfg

def make_sample_sheet(n_per_vowel=10, seed=1):
    df = pd.read_csv(cfg.OUT_DIR / 'tokens_normalised.csv').dropna(subset=['f1', 'f2'])
    parts = [g.sample(min(len(g), n_per_vowel), random_state=seed)
             for _, g in df.groupby('vowel')]
    sheet = pd.concat(parts)[['recording', 'start', 'end', 'vowel', 'word']]
    sheet['f1_manual'] = ''
    sheet['f2_manual'] = ''
    sheet.to_csv(cfg.OUT_DIR / 'manual_sample.csv', index=False)

def validate(manual_csv):
    auto = pd.read_csv(cfg.OUT_DIR / 'tokens_normalised.csv')
    man = pd.read_csv(manual_csv)
    for d in (auto, man):
        d['start_r'] = d['start'].round(3)
    m = auto.merge(man[['recording', 'start_r', 'f1_manual', 'f2_manual']],
                   on=['recording', 'start_r'], how='inner')
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, f in zip(axes, ('f1', 'f2')):
        d = m.dropna(subset=[f, f + '_manual'])
        diff = d[f] - d[f + '_manual']
        ape = diff.abs() / d[f + '_manual'] * 100
        sd = diff.std(ddof=1)
        rows.append({'formant': f.upper(), 'n': len(d),
                     'median_APE_pct': ape.median(), 'mean_APE_pct': ape.mean(),
                     'bias_Hz': diff.mean(),
                     'LoA_lower_Hz': diff.mean() - 1.96 * sd,
                     'LoA_upper_Hz': diff.mean() + 1.96 * sd})
        ax.scatter((d[f] + d[f + '_manual']) / 2, diff, s=10, alpha=0.5)
        for y in (diff.mean(), diff.mean() - 1.96 * sd, diff.mean() + 1.96 * sd):
            ax.axhline(y, linestyle='--', linewidth=1)
        ax.set_xlabel(f.upper() + ' mean of methods (Hz)')
        ax.set_ylabel('Automatic minus manual (Hz)')
    fig.tight_layout()
    fig.savefig(cfg.OUT_DIR / 'bland_altman.png', dpi=300)
    res = pd.DataFrame(rows)
    res.to_csv(cfg.OUT_DIR / 'validation.csv', index=False)
    print(res.round(2).to_string(index=False))

if __name__ == '__main__':
    validate(sys.argv[1])
