
import json


def get_final_phonology(phon):
    rhyme = []
    for ph in phon.split():
        if ph.endswith('1'):
            if rhyme:
                raise ValueError
            else:
                rhyme.append(ph)
        else:
            if rhyme:
                rhyme.append(ph)

    # remove post-stress consontans for monosyllables
    # up/cut, know/coast, time/high, etc...
    nsylls = len([ph for ph in phon.split() if ph[-1].isdigit()])
    if nsylls == 1:
        rhyme = [ph for ph in rhyme if ph[-1].isdigit()]

    return rhyme


def get_rhyme(line1, line2, d):
    try:
        last1 = get_final_phonology(d[line1[-1]])
        last2 = get_final_phonology(d[line2[-1]])
    except (KeyError, ValueError):
        return
    if last1 == last2 and last1 is not None:
        return last1


def get_features(sample, phon_dict, verbose=False):
    nrhymes = 0
    nsents = 0
    rhymes = []

    for l1, l2 in zip(sample[:-1], sample[1:]):
        nsents += 1
        rhyme = get_rhyme(l1, l2, phon_dict)
        if rhyme:
            rhymes.append((rhyme, ' '.join(l1) + '\n' + ' '.join(l2)))
            nrhymes += 1

    if nrhymes:
        if verbose:
            for r, lines in rhymes:
                print(r)
                print(lines)
                print()

            print("---")

    return {'rhyme_density': (nrhymes / nsents) if nsents else 0}


def get_rhymes(sample, phon_dict):
    rhymes, words = [], []
    for l1, l2 in zip(sample[:-1], sample[1:]):
        rhyme = get_rhyme(l1, l2, phon_dict)
        if rhyme:
            rhymes.append('-'.join(rhyme))
            words.append(tuple(sorted([l1[-1], l2[-1]])))
    return list(zip(rhymes, words))


def load_samples():
    import pandas as pd
    import loaders
    phon_dict = loaders.load_phon_dict("sources/ohhla.vocab.phon.json")
    samples = []
    df = pd.read_csv("./sources/db_data.csv")
    for _, row in df.iterrows():
        if not pd.isna(row['real']):
            lines = [s.split() for s in row['real'].split('\n')]
            for rhyme, words in get_rhymes(lines, phon_dict):
                samples.append({'rhyme': rhyme, 'model': 'real', 'words': words})
        if not pd.isna(row['fake']):
            lines = [s.split() for s in row['fake'].split('\n')]
            for rhyme, words in get_rhymes(lines, phon_dict):
                samples.append({'rhyme': rhyme, 'words': words,
                                'model': row['genlevel'],
                                'conditional': row['conditional']})

    return pd.DataFrame.from_dict(samples)


def get_rhyme_dict(path):
    import loaders

    phon_dict = loaders.load_phon_dict("sources/ohhla.vocab.phon.json")

    with open(path) as f:
        for line in f:
            song = json.loads(line.strip())
            for verse in song['text']:
                verse = [[w['token'] for w in line] for line in verse]
                for phon, pair in get_rhymes(verse, phon_dict):
                    yield phon, pair


if __name__ == '__main__':
    # r = load_samples()
    import collections
    rhymes = collections.Counter(get_rhyme_dict('../../deepflow/data/ohhla-new.sorted.train.jsonl'))
    with open('ohhla.rhyme.counts.csv', 'w') as f:
        for (phon, pair), count in rhymes.items():
            f.write('\t'.join([' '.join(pair), phon, str(count)]))
