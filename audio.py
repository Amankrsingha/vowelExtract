# vowelpipe/audio.py
from dataclasses import dataclass
from pathlib import Path
import parselmouth

class AudioError(ValueError):
    '''Raised when a recording cannot be analysed.'''

@dataclass
class AudioInfo:
    filename: str
    duration: float
    sample_rate: float
    n_channels: int

def load_sound(path: Path, ceiling: float = 5500.0, min_duration: float = 0.5):
    '''Load a WAV file, validate it and return (mono Sound, AudioInfo).'''
    try:
        snd = parselmouth.Sound(str(path))
    except Exception as exc:
        raise AudioError(f'Cannot read {path.name}: {exc}') from exc

    # Record the original properties before any conversion.
    info = AudioInfo(path.name, snd.duration, snd.sampling_frequency, snd.n_channels)

    if snd.n_channels > 1:
        snd = snd.convert_to_mono()
    if snd.sampling_frequency < 2 * ceiling:
        raise AudioError(
            f'{path.name}: sampling rate {snd.sampling_frequency:.0f} Hz is too low '
            f'for a {ceiling:.0f} Hz formant ceiling (needs at least {2 * ceiling:.0f} Hz).')
    if snd.duration < min_duration:
        raise AudioError(f'{path.name}: recording is shorter than {min_duration} s.')
    if snd.get_rms() < 1e-6:
        raise AudioError(f'{path.name}: recording appears to be silent.')
    return snd, info
