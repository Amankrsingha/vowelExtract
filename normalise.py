# vowelpipe/normalise.py
import numpy as np
import pandas as pd

FORMANTS = ('f1', 'f2')

def hz_to_bark(f):
    f = np.asarray(f, dtype=float)
    return 26.81 * f / (1960.0 + f) - 0.53          # Traunmueller (1990)

def hz_to_mel(f):
    f = np.asarray(f, dtype=float)
    return 2595.0 * np.log10(1.0 + f / 700.0)

def add_scales(df, cols=FORMANTS):
    out = df.copy()
    for c in cols:
        out[f'{c}_bark'] = hz_to_bark(out[c])
        out[f'{c}_mel'] = hz_to_mel(out[c])
    return out

def lobanov(df, cols=FORMANTS, by='speaker'):
    '''z-score each formant within each speaker (Lobanov, 1971).'''
    out = df.copy()
    for c in cols:
        g = out.groupby(by)[c]
        out[f'{c}_lob'] = (out[c] - g.transform('mean')) / g.transform('std')
    return out

def nearey1(df, cols=FORMANTS, by='speaker'):
    '''Subtract each speaker's mean log formant (Nearey, 1978).'''
    out = df.copy()
    for c in cols:
        logf = np.log(out[c])
        out[f'{c}_n1'] = logf - logf.groupby(out[by]).transform('mean')
    return out

def normalise_all(df, min_tokens=30):
    '''Keep QC-passed tokens, drop thin speakers, add all scales/normalisations.'''
    d = df[df['keep']].copy()
    n = d.groupby('speaker')['f1'].transform('count')
    d = d[n >= min_tokens]
    return nearey1(lobanov(add_scales(d)))
# vowelpipe/normalise.py (continued)
from scipy.stats import chi2
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

def ellipse_params(x, y, coverage=0.95):
    '''Centre, axis lengths and rotation (degrees) of a covariance ellipse.'''
    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    if len(pts) < 3:
        return None
    mu = pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(pts, rowvar=False))
    order = np.argsort(vals)[::-1]                 # largest eigenvalue first
    vals, vecs = vals[order], vecs[:, order]
    k = np.sqrt(chi2.ppf(coverage, df=2))
    return {'cx': float(mu[0]), 'cy': float(mu[1]),
            'width': float(2 * k * np.sqrt(vals[0])),
            'height': float(2 * k * np.sqrt(vals[1])),
            'angle': float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))}

def category_summary(df, cols):
    return df.groupby('vowel')[list(cols)].agg(['mean', 'std', 'count'])

def category_means(df, xcol, ycol):
    return df.groupby('vowel')[[xcol, ycol]].mean()

def vowel_space_area(means):
    '''Area of the convex hull of the vowel means (needs at least 3 vowels).'''
    pts = means.to_numpy()
    return float(ConvexHull(pts).volume) if len(pts) >= 3 else float('nan')

def mean_distance_matrix(means):
    d = squareform(pdist(means.to_numpy()))
    return pd.DataFrame(d, index=means.index, columns=means.index)
# vowelpipe/normalise.py (continued)
from statsmodels.multivariate.manova import MANOVA

METHODS = {'hz': ('f1', 'f2'), 'bark': ('f1_bark', 'f2_bark'),
           'mel': ('f1_mel', 'f2_mel'), 'lobanov': ('f1_lob', 'f2_lob'),
           'nearey1': ('f1_n1', 'f2_n1')}

def sex_gap(df, col):
    '''Mean standardised female-male difference in speaker vowel means.'''
    sp = df.groupby(['speaker', 'sex', 'vowel'])[col].mean().reset_index()
    gaps = []
    for _, g in sp.groupby('vowel'):
        a = g.loc[g['sex'] == 'female', col]
        b = g.loc[g['sex'] == 'male', col]
        if len(a) < 2 or len(b) < 2:
            continue
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        gaps.append(abs(a.mean() - b.mean()) / sd)
    return float(np.mean(gaps)) if gaps else float('nan')

def pillai(df, f1col, f2col):
    res = MANOVA.from_formula(f'{f1col} + {f2col} ~ vowel', data=df).mv_test()
    return float(res.results['vowel']['stat'].loc["Pillai's trace", 'Value'])

def compare_methods(df):
    rows = []
    for name, (a, b) in METHODS.items():
        rows.append({'method': name, 'sex_gap_F1': sex_gap(df, a),
                     'sex_gap_F2': sex_gap(df, b), 'pillai': pillai(df, a, b)})
    return pd.DataFrame(rows)

