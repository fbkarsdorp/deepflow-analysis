
import csv
import json


PATHS = {
    'dbpath': 'deepflow_db.dump.jsonl',
    'spath': 'samples.combined.jsonl',
    'tpath': 'turing-pairs.jsonl',
    'outputpath': 'db_data.csv'
}

entries = []
samples = {}
pairs = {}


with open(PATHS['dbpath']) as f:
    for l in f:
        entries.append(json.loads(l))

with open(PATHS['spath']) as f:
    for l in f:
        obj = json.loads(l)
        samples[obj['id']] = obj

with open(PATHS['tpath']) as f:
    for l in f:
        obj = json.loads(l)
        pairs[obj['id']] = obj


def create_id2pairid(output='id2pairid.csv'):
    # # Create a dict from sample to pair id
    with open(PATHS['tpath']) as f, open(output, 'w') as out:
        out.write("{}\t{}\t{}\n".format('pair', 'true', 'fake'))
        for l in f:
            obj = json.loads(l)
            out.write("{}\t{}\t{}\n".format(obj['id'], obj['true_id'], obj['false_id']))


def get_times(test):
    times = []
    last, selected, started = None, False, False
    for log in test['log']['log']:
        if log['type'] == 'select':
            if last is None:
                raise ValueError("Got select but didn't receive question yet")
            if selected:        # overwrite with new selection
                times[-1] = log['timestamp'] - last['timestamp']
            else:
                selected = True
                times.append(log['timestamp'] - last['timestamp'])
        elif log['type'] == 'new question received':
            if not selected and started:  # timeout
                times.append(float('inf'))
                # times.append(log['timestamp'] - last['timestamp'])
            last = log
            selected = False
            started = True
        else:
            pass
    # add last if not selected
    if not selected:
        times.append(float('inf'))
    return times


rows = []

for test in entries:

    # get times
    times = get_times(test)

    for idx, q in enumerate(test['log']['questions']):
        if q['raw']['id'] not in pairs:  # couldn't find the samples
            continue

        if pairs[q['raw']['id']]['false_id'] not in samples:
            raise ValueError("No metadata for this sample")

        # can't use this since it gets overwritten when unanswered
        # true_answer = q['answer']
        if q['type'] == 'forreal':
            real = '' if q['line'] == q['raw']['fake'] else '\n'.join(q['line'])
            fake = '' if q['line'] == q['raw']['real'] else '\n'.join(q['line'])
            true_answer = 1 if real else 2
        elif q['type'] == 'choose':
            real = '\n'.join(q['raw']['real'])
            fake = '\n'.join(q['raw']['fake'])
            true_answer = 1 if q['raw']['real'] == q['line1'] else 2

        user_answer = q['selected'] if 'selected' in q else 0

        fake_meta = samples[pairs[q['raw']['id']]['false_id']]
        model = fake_meta['model']
        if 'Char' in model:
            genlevel = 'char'
        elif 'Hybrid' in model:
            genlevel = 'hybrid'
        else:
            genlevel = 'syl'
        conditional = model in ('CharLanguageModel.2018-08-06+17:01:47',
                                'HybridLanguageModel.2018-08-15+12:35:02',
                                'RNNLanguageModel.2018-08-01+19:33:04')

        rows.append({
            # test
            'test_id': test['id'],
            'score': test['score'],
            # question
            'iteration': q['raw']['iteration'],
            'level': q['raw']['level'],
            'type': q['type'],
            'pair_id': q['raw']['id'],
            'real': real,
            'fake': fake,
            'true_answer': true_answer,
            'user_answer': user_answer,
            'correct': true_answer == user_answer,
            'time': times[idx],
            # model
            'tau': fake_meta['params']['tau'],
            'model': model,
            'genlevel': genlevel,
            'conditional': conditional
        })

# check all times are equal in length to questions
for idx, test in enumerate(entries):
    times = get_times(test)
    assert len(times) == len(test['log']['questions']), \
        (idx, times, len(test['log']['questions']))

# check that unanswered questions correspond to "selected" missing
for idx, test in enumerate(entries):
    times = get_times(test)
    for time, q in zip(times, test['log']['questions']):
        if time < float('inf'):
            assert 'selected' in q
        else:
            assert 'selected' not in q

# write to file
with open(PATHS['outputpath'], 'w') as f:
    writer = csv.DictWriter(f, rows[0].keys())
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
