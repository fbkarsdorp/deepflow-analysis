

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
