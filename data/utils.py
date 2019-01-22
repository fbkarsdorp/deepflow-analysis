
from syllabify import syllabify


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
