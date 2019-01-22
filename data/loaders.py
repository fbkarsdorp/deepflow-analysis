
import json
import collections
import syntactic_features


# LOADERS
def load_function(path):
    with open(path) as f:
        return set(w.strip() for w in f)
    
def load_phon_dict(path):
    "phon_dict_path"
    with open(path) as f:
        phon_dict = json.load(f)
    return phon_dict


def load_pc_words(path):
    "pc_wordspath"
    with open(path) as f:
        pc_words = json.load(f)['words']
    return pc_words


def load_freqs(path):
    "freqspath"
    freqs = {}
    with open(path) as f:
        for line in f:
            w, freq = line.strip().split()
            freqs[w] = float(freq)
    return freqs


def load_pronouns(path):
    "pronpath"
    prons = []
    with open(path) as f:
        for line in f:
            prons.append(line.strip())
    return set(prons)


def create_ohhla_freqs(path, outputpath="freqs.csv"):
    "Create freqs.csv general word frequencies from ohhla (ohhla path)"
    words = collections.Counter()
    with open(path) as f:
        for line in f:
            song = json.loads(line.strip())
            for verse in song['text']:
                for line in verse:
                    for w in line:
                        words[w['token']] += 1

    with open(outputpath, 'w') as f:
        for w, c in words.most_common():
            f.write('{}\t{}\n'.format(w, c))
