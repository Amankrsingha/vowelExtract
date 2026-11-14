# vowelpipe/textgrid_io.py
import re
import parselmouth
from parselmouth.praat import call

class AlignmentError(ValueError):
    '''Raised when a TextGrid does not match its recording.'''

def read_textgrid(path):
    return parselmouth.read(str(path))

def tier_names(tg):
    n = int(call(tg, 'Get number of tiers'))
    return [call(tg, 'Get tier name', i) for i in range(1, n + 1)]

def iter_intervals(tg, tier):
    '''Yield (index, start, end, label) for every interval on a tier.'''
    n = int(call(tg, 'Get number of intervals', tier))
    for i in range(1, n + 1):
        yield (i,
               call(tg, 'Get start time of interval', tier, i),
               call(tg, 'Get end time of interval', tier, i),
               call(tg, 'Get label of interval', tier, i))

def clean_label(label: str) -> str:
    '''Remove whitespace and a trailing stress digit: 'AE1' -> 'AE'.'''
    return re.sub('[0-9]+$', '', label.strip())

def check_alignment(tg, sound_duration: float, tol: float = 0.05):
    tg_end = call(tg, 'Get end time')
    if abs(tg_end - sound_duration) > tol:
        raise AlignmentError(
            f'TextGrid ends at {tg_end:.3f} s but the audio lasts {sound_duration:.3f} s.')

def vowel_intervals(tg, phone_tier, word_tier, vowels, min_dur):
    '''Return one dict per vowel interval, with word and context phones.'''
    intervals = list(iter_intervals(tg, phone_tier))
    rows = []
    for k, (i, t0, t1, lab) in enumerate(intervals):
        vowel = clean_label(lab)
        if vowel not in vowels or (t1 - t0) < min_dur:
            continue
        prev_ph = clean_label(intervals[k - 1][3]) if k > 0 else ''
        next_ph = clean_label(intervals[k + 1][3]) if k < len(intervals) - 1 else ''
        word = ''
        if word_tier:
            mid = 0.5 * (t0 + t1)
            wi = int(call(tg, 'Get interval at time', word_tier, mid))
            if wi > 0:
                word = call(tg, 'Get label of interval', word_tier, wi).strip().lower()
        rows.append({'vowel': vowel, 'label_raw': lab.strip(), 'start': t0, 'end': t1,
                     'duration': t1 - t0, 'word': word,
                     'prev_phone': prev_ph, 'next_phone': next_ph})
    return rows
