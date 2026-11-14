# vowelpipe/run_pipeline.py       run with:  python -m vowelpipe.run_pipeline
import json
import logging
import sys
import time
from collections import defaultdict
import numpy
import pandas as pd
import parselmouth
import scipy
import sqlalchemy
from . import config as cfg
from .audio import load_sound
from .textgrid_io import read_textgrid, check_alignment, vowel_intervals
from .formants import extract_tokens, optimise_ceiling, apply_qc
from .normalise import normalise_all, compare_methods, category_means, vowel_space_area
from .plots import static_vowel_space, interactive_vowel_space
from .db import default_engine, ingest

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
log = logging.getLogger('pipeline')

NORM_COLUMNS = {'lobanov': ('f1_lob', 'f2_lob'), 'nearey1': ('f1_n1', 'f2_n1'),
                'bark': ('f1_bark', 'f2_bark'), 'mel': ('f1_mel', 'f2_mel')}

def params_dict():
    keys = ['PHONE_TIER', 'WORD_TIER', 'VOWELS', 'MIN_VOWEL_DUR', 'ALIGN_TOLERANCE',
            'MAX_FORMANT', 'OPTIMISE_CEILING', 'CEILING_CANDIDATES', 'N_FORMANTS',
            'WINDOW', 'PREEMPH', 'TIME_STEP', 'SAMPLE_POINTS', 'PITCH_RANGE',
            'F1_RANGE', 'F2_RANGE', 'OUTLIER_Z', 'OUTLIER_MIN_N']
    d = {k: getattr(cfg, k) for k in keys}
    return json.loads(json.dumps(d, default=sorted))      # sets -> sorted lists

def versions():
    return {'python': sys.version.split()[0], 'parselmouth': parselmouth.VERSION,
            'praat': parselmouth.PRAAT_VERSION, 'numpy': numpy.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__,
            'sqlalchemy': sqlalchemy.__version__}

def main():
    t_total = time.perf_counter()
    cfg.OUT_DIR.mkdir(exist_ok=True)
    speakers = pd.read_csv(cfg.SPEAKER_CSV, dtype=str).fillna('')
    sex_of = dict(zip(speakers['speaker'], speakers['sex']))

    # ---- Stage 1: load, validate and segment (Section 4.1) ----
    items = defaultdict(list)            # speaker -> [(filename, Sound, tokens)]
    rec_rows, problems, total_audio = [], [], 0.0
    wavs = sorted(p for p in cfg.WAV_DIR.iterdir() if p.suffix.lower() == '.wav')
    for wav in wavs:
        spk = wav.stem.split('_')[0]
        tg_path = cfg.TG_DIR / (wav.stem + '.TextGrid')
        if spk not in sex_of:
            problems.append((wav.name, 'speaker not listed in speakers.csv'))
            continue
        if not tg_path.exists():
            problems.append((wav.name, 'no matching TextGrid'))
            continue
        try:
            snd, info = load_sound(wav, ceiling=cfg.MAX_FORMANT[sex_of[spk]])
            tg = read_textgrid(tg_path)
            check_alignment(tg, info.duration, cfg.ALIGN_TOLERANCE)
            tokens = vowel_intervals(tg, cfg.PHONE_TIER, cfg.WORD_TIER,
                                     cfg.VOWELS, cfg.MIN_VOWEL_DUR)
        except Exception as exc:         # one bad file must not stop the batch
            problems.append((wav.name, str(exc)))
            continue
        items[spk].append((wav.name, snd, tokens))
        rec_rows.append({'speaker': spk, 'filename': wav.name,
                         'duration': info.duration, 'sample_rate': info.sample_rate,
                         'n_channels': info.n_channels, 'textgrid': tg_path.name})
        total_audio += info.duration
    log.info('%d recordings accepted, %d problems', len(rec_rows), len(problems))

    # ---- Stage 2: formant extraction and QC (Section 4.2) ----
    t_extract = time.perf_counter()
    frames = []
    for spk, recs in items.items():
        sex = sex_of[spk]
        ceiling = cfg.MAX_FORMANT[sex]
        if cfg.OPTIMISE_CEILING:
            best = optimise_ceiling([(s, t) for _, s, t in recs], cfg.CEILING_CANDIDATES)
            ceiling = best or ceiling
        log.info('speaker %s (%s): ceiling %.0f Hz', spk, sex, ceiling)
        for name, snd, tokens in recs:
            df = extract_tokens(snd, tokens, ceiling, sex)
            if df.empty:
                continue
            df['speaker'], df['recording'], df['sex'] = spk, name, sex
            frames.append(df)
    extract_seconds = time.perf_counter() - t_extract
    raw = apply_qc(pd.concat(frames, ignore_index=True))
    raw = raw.merge(speakers[['speaker', 'dialect', 'age_group']], on='speaker', how='left')

    # ---- Stage 3: normalisation and comparison (Section 4.3) ----
    norm = normalise_all(raw)
    comparison = compare_methods(norm)
    new_cols = [c for c in norm.columns if c not in raw.columns]
    full = raw.join(norm[new_cols])      # excluded tokens receive missing values

    # ---- Stage 4: tables, figures and database (Section 4.4) ----
    out = cfg.OUT_DIR
    full.to_csv(out / 'tokens_normalised.csv', index=False)
    comparison.to_csv(out / 'normalisation_comparison.csv', index=False)
    raw.groupby(['speaker', 'qc']).size().unstack(fill_value=0).to_csv(out / 'qc_counts.csv')
    pd.DataFrame(problems, columns=['file', 'problem']).to_csv(out / 'problems.csv', index=False)
    (norm.groupby(['sex', 'vowel'])[['f1', 'f2', 'f1_lob', 'f2_lob']]
         .agg(['mean', 'std', 'count']).to_csv(out / 'group_summary.csv'))

    areas = []
    for spk, g in norm.groupby('speaker'):
        means = category_means(g, 'f2_lob', 'f1_lob')
        areas.append({'speaker': spk, 'sex': g['sex'].iloc[0],
                      'n_vowels': len(means), 'area_lobanov': vowel_space_area(means)})
    pd.DataFrame(areas).to_csv(out / 'speaker_areas.csv', index=False)

    for sex, g in norm.groupby('sex'):
        static_vowel_space(g, 'f2', 'f1', out / f'vowel_space_hz_{sex}.png',
                           xlabel='F2 (Hz)', ylabel='F1 (Hz)')
    static_vowel_space(norm, 'f2_lob', 'f1_lob', out / 'vowel_space_lobanov.png',
                       xlabel='F2 (z-score)', ylabel='F1 (z-score)')
    interactive_vowel_space(norm, 'f2_lob', 'f1_lob', out / 'vowel_space_interactive.html')

    run_id = ingest(default_engine(), speakers, pd.DataFrame(rec_rows), full,
                    NORM_COLUMNS, params_dict(), versions())

    stats = {'run_id': run_id, 'n_recordings': len(rec_rows), 'n_tokens': int(len(raw)),
             'n_tokens_kept': int(raw['keep'].sum()), 'audio_seconds': total_audio,
             'extraction_seconds': extract_seconds,
             'real_time_factor': extract_seconds / total_audio,
             'total_seconds': time.perf_counter() - t_total,
             'params': params_dict(), 'versions': versions()}
    (out / 'run_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    log.info('done: run %d, real-time factor %.3f', run_id, stats['real_time_factor'])

if __name__ == '__main__':
    main()
