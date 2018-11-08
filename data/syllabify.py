
import os
from allennlp.predictors import Predictor


path = 'sources/syllable-model.tar.gz'
if not os.path.isfile(path):
    raise FileNotFoundError(path)

syllabifier = Predictor.from_path(path)


def syllabify(words):
    """
    Transform into syllables

    Arguments:
    - syllabifier : Predictor
    - words : list of str
    """
    sent = []
    for word in words:
        syllables = []
        pred = syllabifier.predict(' '.join(word))
        for char, tag in zip(pred['words'], pred['tags']):
            if int(tag) > 0:
                syllables.append('')
            syllables[-1] += char

        sent.extend(syllables)

    return sent
