# vowelpipe/formants.py
import math
import pandas as pd
from . import config as cfg

def _clean(x):
    '''Praat returns NaN when a value is undefined; keep that as None.'''
    x = float(x)
    return None if math.isnan(x) else x

def make_formant(snd, ceiling):
    return snd.to_formant_burg(
        time_step=cfg.TIME_STEP,
        max_number_of_formants=cfg.N_FORMANTS,
        maximum_formant=ceiling,
        window_length=cfg.WINDOW,
        pre_emphasis_from=cfg.PREEMPH)

def extract_tokens(snd, tokens, ceiling, sex):
    '''Measure F1-F3 at each sample point, plus f0 at the midpoint.'''
    if not tokens:
        return pd.DataFrame()
    formant = make_formant(snd, ceiling)
    lo, hi = cfg.PITCH_RANGE[sex]
    pitch = snd.to_pitch(pitch_floor=lo, pitch_ceiling=hi)

    rows = []
    for tok in tokens:
        rec = dict(tok)
        dur = tok['end'] - tok['start']
        for p in cfg.SAMPLE_POINTS:
            t = tok['start'] + p * dur
            tag = int(round(p * 100))
            for k in (1, 2, 3):
                rec[f'f{k}_{tag}'] = _clean(formant.get_value_at_time(k, t))
        mid = tok['start'] + 0.5 * dur
        rec['f0'] = _clean(pitch.get_value_at_time(mid))
        rec['ceiling_hz'] = ceiling
        rows.append(rec)

    df = pd.DataFrame(rows)
    for k in (1, 2, 3):                 # the midpoint is the primary measurement
        df[f'f{k}'] = df[f'f{k}_50']
    return df
# vowelpipe/formants.py (continued)
def optimise_ceiling(items, candidates):
    '''items: list of (Sound, tokens) for ONE speaker.
    Returns the candidate ceiling with the smallest mean within-vowel
    variance of log F1 and log F2, or None if there are too few tokens.'''
    best, best_score = None, float('inf')
    for c in candidates:
        rows = []
        for snd, tokens in items:
            fm = make_formant(snd, c)
            for tok in tokens:
                t = 0.5 * (tok['start'] + tok['end'])
                f1 = fm.get_value_at_time(1, t)
                f2 = fm.get_value_at_time(2, t)
                if not (math.isnan(f1) or math.isnan(f2)):
                    rows.append((tok['vowel'], math.log(f1), math.log(f2)))
        if len(rows) < 30:
            continue
        d = pd.DataFrame(rows, columns=['vowel', 'l1', 'l2'])
        v = d.groupby('vowel')[['l1', 'l2']].var().dropna()
        score = float(v.sum(axis=1).mean())
        if score < best_score:
            best, best_score = c, score
    return best
# vowelpipe/formants.py (continued)
def apply_qc(df):
    df = df.copy()
    df['qc'] = ''
    df.loc[df[['f1', 'f2']].isna().any(axis=1), 'qc'] = 'undefined'

    ok = df['qc'] == ''
    bad = ~df['f1'].between(*cfg.F1_RANGE) | ~df['f2'].between(*cfg.F2_RANGE) | (df['f1'] >= df['f2'])
    df.loc[ok & bad, 'qc'] = 'implausible'

    sub = df[df['qc'] == '']
    g = sub.groupby(['speaker', 'vowel'])
    for col in ('f1', 'f2'):
        z = (sub[col] - g[col].transform('mean')) / g[col].transform('std')
        n = g[col].transform('count')
        flagged = sub.index[(z.abs() > cfg.OUTLIER_Z) & (n >= cfg.OUTLIER_MIN_N)]
        df.loc[flagged, 'qc'] = 'outlier'

    df['keep'] = df['qc'] == ''
    return df
