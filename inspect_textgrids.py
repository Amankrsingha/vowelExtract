# vowelpipe/inspect_textgrids.py   run with:  python -m vowelpipe.inspect_textgrids
from collections import Counter
from . import config as cfg
from .textgrid_io import read_textgrid, tier_names, iter_intervals

def main():
    labels = Counter()
    for path in sorted(cfg.TG_DIR.glob('*.TextGrid')):
        tg = read_textgrid(path)
        print(path.name, tier_names(tg))
        for _, _, _, lab in iter_intervals(tg, cfg.PHONE_TIER):
            labels[lab.strip()] += 1
    print()
    print('Labels on tier', cfg.PHONE_TIER, '(most frequent first):')
    for lab, n in labels.most_common():
        print(repr(lab), n)

if __name__ == '__main__':
    main()
