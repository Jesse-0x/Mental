import json

dataset = json.load(open('ESConv2.json'))

cate = []
# load each dialog to see the supporter's category
for i in dataset:
    for j in i['dialog']:
        if 'strategy' in j['annotation'].keys():
            cate.append(j['annotation']['strategy'])

type_count = {}
for i in cate:
    type_count[i] = 0
# for each category, find 500 dialogs example

import sqlite3
import random

conn = sqlite3.connect('ESConv.db')
ids = []
new_dataset = []
evaluation_ids = []
all_dialog = []
for i in range(5):
    c = conn.cursor()
    c.execute('''SELECT * FROM ESConv WHERE seeker_survey_iei - seeker_survey_fei = ?''', (4-i,))
    conn.commit()
    rt = c.fetchall()
    ids = [i[-1] for i in rt]
    print(ids)
    for i in dataset:
        for j in i['dialog']:
            if i['id'] in ids:
                all_dialog.append(j)
    all_dialog.sort(key=lambda x: len(x['content']), reverse=True)
    for j in all_dialog:
        if 'strategy' in j['annotation'].keys():
            # for each type in cates, get 13 examples
            if type_count[j['annotation']['strategy']] < 13:
                type_count[j['annotation']['strategy']] += 1
                evaluation_ids.append(j['id'])
                print('add', end='')

with open('evaluation_ids2.json', 'w') as outfile:
    json.dump(evaluation_ids, outfile, indent=4)