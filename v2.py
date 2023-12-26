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
for i in range(5):
    c = conn.cursor()
    c.execute('''SELECT * FROM ESConv WHERE seeker_survey_iei - seeker_survey_fei = ?''', (i,))
    conn.commit()
    rt = c.fetchall()
    tids = [i[-1] for i in rt]
    ids += tids
new_dataset = []
evaluation_ids = []
for i in ids:
    cur = [j for j in dataset if j['id'] == i][0]
    for j in cur['dialog']:
        if 'strategy' in j['annotation'].keys():
            # for each type in cates, get 13 examples
            if type_count[j['annotation']['strategy']] < 13:
                type_count[j['annotation']['strategy']] += 1
                evaluation_ids.append(j['id'])
