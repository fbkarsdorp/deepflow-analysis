
import os
import json
import itertools
import pandas as pd
import numpy as np
import scipy.stats

import loaders
import lzw
from syllabify import syllabify
import syntactic_features


PATHS = {
    'phon_dict': 'ohhla.vocab.phon.json',
    'pc_words': 'censuur.json',
    'freqs': 'freqs.csv',
    'pronouns': 'pronouns.txt',
    'syllabifier': 'syllable-model.tar.gz',
    'pairs': 'turing-pairs.jsonl',
    'samples': 'samples.combined.jsonl',
    'db': 'db_data.csv'
}

MODELDATA = {
    'CharLanguageModel.2018-08-06+17:01:47': ('char', 'conditional'),
    'HybridLanguageModel.2018-08-15+12:35:02': ('hierarchical', 'conditional'),
    'RNNLanguageModel.2018-08-01+19:33:04': ('word', 'conditional'),
    'CharLanguageModel.2018-08-13+17:31:16': ('char', 'inconditional'),
    'HybridLanguageModel.2018-08-15+14:33:35': ('hierarchical', 'inconditional'),
    'RNNLanguageModel.2018-08-13+10:23:55': ('word', 'inconditional'),
    None: ('real', 'whateva')
}


VOWELS = 'AA AE AH AO AW AX AY EH ER EY IH IX IY OW OY UH UW UX'.split()


def entropy(items, base=None):
    _, counts = np.lib.arraysetops.unique(list(filter(None, items)), return_counts=True)
    return scipy.stats.entropy(counts)


def word_entropy(lines, freqs, total):
    probs = np.array([freqs.get(w, 1e-20)/total for line in lines for w in line])
    return -(probs*np.log(probs)).sum()


def assonance_entropy(lines, phon_dict):
    def stressed_nucleus(word):
        phones = phon_dict.get(word, '')
        return next((p for p in phones.split()[::-1] if p[-1] == "1"), None)

    scores = [entropy(stressed_nucleus(w) for w in line) for line in lines]
    return sum(scores) / len(scores)


def onset(word, phon_dict):
    def strip_stress(p): return p[:-1] if p.endswith(('1', '2')) else p
    phones = phon_dict.get(word, None)
    if phones is not None:
        phones = ''.join(itertools.takewhile(
            lambda p: strip_stress(p) not in VOWELS, phones.split()))
    return phones


def onset_entropy(lines, phon_dict):
    scores = [entropy(onset(w, phon_dict) for w in line) for line in lines]
    return sum(scores) / len(scores)


def alliteration_score(lines, phon_dict):
    alliterations = 0
    for line in lines:
        for a, b in zip(line, line[1:]):
            onset_a, onset_b = onset(a, phon_dict), onset(b, phon_dict)
            if onset_a is not None and onset_a == onset_b:
                alliterations += 1
    return alliterations / len(sum(lines, []))


def vocab_entropy(lines):
    return entropy(word for line in lines for word in line)


# UTILITY
def group_syllables(line):
    words = []
    for syllable in line:
        if syllable.startswith("-"):
            if not words:
                words.append([])
            words[-1].append(syllable)
        else:
            words.append([syllable])
    return words


def format_word(syllables):
    if len(syllables) == 1:
        return syllables[0]

    def strip_hyphens(syllable):
        return syllable[syllable.startswith("-"):-syllable.endswith("-") or None]

    return "".join(strip_hyphens(syllable) for syllable in syllables)


def read_sample(sample, words=False, source='original'):
    lines = [line[source].split() for line in sample["text"]]

    # character-level models: char-level was trained on free running text
    # (so "original" doesn't have syllable boundaries)
    if sample.get('model', 'default').lower().startswith('char'):
        if words:
            return lines
        return [syllabify(line) for line in lines]

    # all other cases
    if words:
        lines = [[format_word(sylls) for sylls in group_syllables(line)]
                 for line in lines]

    return lines


def extract_features(sample, phon_dict, pc_words, prons, freqs, total_freqs):
    features = {}
    sylls = read_sample(sample)
    words = read_sample(sample, words=True)
    nwords = sum(len(line) for line in words)

    # average word length
    features['word-length'] = np.mean([len(w) for line in words for w in line])
    # average word length in syllables
    features['word-length-syllables'] = np.mean([
        len(s) for line in sylls for s in group_syllables(line)])
    # alliteration score
    features['alliteration'] = alliteration_score(words, phon_dict)
    # repetitiveness
    features['stressed-vowel-repetitiveness'] = assonance_entropy(
        words, phon_dict)
    features['word-onset-repetitiveness'] = onset_entropy(words, phon_dict)
    features['word-repetitiveness'] = vocab_entropy(words)
    features['syllable-repetitiveness'] = vocab_entropy(sylls)
    # proportion of pc words in line
    features['pc-words'] = \
        len([w for line in words for w in line if w in pc_words]) / nwords
    features['pronouns'] = \
        len([w for line in words for w in line if w in prons]) / nwords
    # lexical diversity wrt general corpus
    features['word-entropy'] = word_entropy(words, freqs, total_freqs)
    features['line'] = words
    features['lzw'] = len(' '.join([w for l in words for w in l])) / \
        len(lzw.compress(' '.join([w for l in words for w in l])))
    # syntactic features
    features['nwords'] = nwords
    features['nchars'] = sum(len(w) for line in words for w in line)
    features['nlines'] = len(words)
    sentences = [' '.join(s) for s in read_sample(sample, words=True, source='line')]
    for key, val in syntactic_features.get_features(sentences).items():
        features[key] = val

    return features


def preprocess_db(dbpath, pairspath, samplespath,
                  phon_dict, pc_words, prons, freqs, total_freqs):
    samples = []
    # read info from db_data by pair id
    db_pair = {}
    db_keys = 'score iteration level type pair_id true_answer user_answer correct time'
    for _, row in pd.DataFrame.from_csv(dbpath).iterrows():
        db_pair[row['pair_id']] = {key: row[key] for key in db_keys.split()}

    sample2pair = {}   # map sample ids to pair ids

    # get sample ids and pair scores from turing-pairs.jsonl
    with open(pairspath) as f:
        for line in f:
            obj = json.loads(line.strip())
            pair_id = obj['id']
            if pair_id in db_pair:
                # store score assigned by adversary discriminator
                db_pair[pair_id]['pair_score'] = obj['score']
                # store ids
                sample2pair[obj["true_id"]] = obj['id']
                sample2pair[obj["false_id"]] = obj['id']

    done = 0
    with open(samplespath) as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj['id'] not in sample2pair:
                continue

            done += 1
            if done % 100 == 0:
                print(".", end="", flush=True)

            sample = {}
            sample['id'] = obj['id']
            # - db data
            for key, val in db_pair[sample2pair[obj['id']]].items():
                sample[key] = val
            # - sample data
            modeltype, conditional = MODELDATA[sample.get('model')]
            sample['modeltype'] = modeltype
            sample['conditional'] = conditional
            sample['source'] = 'real'
            # (generation params)
            if 'params' in obj:  # add generation metadata
                sample['source'] = 'fake'
                sample['tau'] = obj['params']['tau']
                sample['tries'] = obj['params']['tries']
                sample['model-ppl'] = \
                    sum(l['params']['score'] for l in obj['text']) / len(obj['text'])
            # - feature extraction
            for k, v in extract_features(
                    obj, phon_dict, pc_words, prons, freqs, total_freqs).items():
                sample[k] = v

            samples.append(sample)

    return samples


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # total_samples=5000

    args = parser.parse_args()

    # check paths
    for resource, path in PATHS.items():
        print("Checking", resource, path)
        path = os.path.join('sources', path)
        if not os.path.isfile(path):
            raise ValueError("Couldn't find file [{}]".format(path))
        PATHS[resource] = path
    print(PATHS)

    # required resources
    phon_dict = loaders.load_phon_dict(PATHS['phon_dict'])
    pc_words = loaders.load_pc_words(PATHS['pc_words'])
    prons = loaders.load_pronouns(PATHS['pronouns'])
    freqs = loaders.load_freqs(PATHS['freqs'])
    total_freqs = sum(freqs.values())

    samples = preprocess_db(
        PATHS['db'], PATHS['pairs'], PATHS['samples'],
        phon_dict, pc_words, prons, freqs, total_freqs)

    pd.DataFrame.from_dict(samples).to_csv('db_features.csv')
