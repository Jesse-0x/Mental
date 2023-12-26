import json
print('Loading the models...')
from FID import fid
import tqdm
evaluation = json.load(open("../gpt4/gpt4_output.json"))
dataset = json.load(open("../gpt4/ESConv2.json"))
reference = {
    "Question":[],
    "Self-disclosure":[],
    "Affirmation and Reassurance":[],
    "Providing Suggestions":[],
    "Others":[],
    "Reflection of feelings":[],
    "Information":[],
    "Restatement or Paraphrasing":[],
}
types = {
    "Question":[],
    "Self-disclosure":[],
    "Affirmation and Reassurance":[],
    "Providing Suggestions":[],
    "Others":[],
    "Reflection of feelings":[],
    "Information":[],
    "Restatement or Paraphrasing":[],
}


def find_dialog_type(dialog_id):
    for i in range(len(dataset)):
        for j in range(len(dataset[i]['dialog'])):
            if dataset[i]['dialog'][j]['id'] == dialog_id:
                return dataset[i]['dialog'][j]['annotation']['strategy']
    raise Exception('IDK What the hack is happening here')

print('Start collecting the reference and types...')
for i in tqdm.tqdm(range(len(evaluation))):
    for j in range(len(evaluation[i]['dialog'])):
        if 'response' in evaluation[i]['dialog'][j]:
            strategy = find_dialog_type(evaluation[i]['dialog'][j]['id'])
            reference[strategy].append(evaluation[i]['dialog'][j]['response'])
            # find the dialog id in the dataset
            types[strategy].append(evaluation[i]['dialog'][j]['content'])

print('Start calculating FID')
o_scores = {}
#overall: -3.2882782293517914e+26
for i in types:
    print('-------------------')
    score = fid(reference[i], types[i])
    print(i, score)
    o_scores[i] = score

# {"Question":1.4462000594139885e+39,
# "Self-disclosure ":2.839537344511234e+32,
# "Affirmation and Reassurance":1.298074214633707e+33,
# "Providing Suggestions":3.1691265005705735e+29,
# "Others":8.36826229095459,
# "Reflection of feelings ":6.084722881095501e+32,
# "Information":8.307674973655724e+35,
# "Restatement or Paraphrasing":1.5548526893424376e+30,}

import torch
from torch.nn.functional import softmax
import numpy as np

import torch
from torch.nn.functional import softmax
import numpy as np

data = torch.tensor(list(o_scores.values()), dtype=torch.float64)

# Normalize data
mean = torch.mean(data)
std = torch.std(data)
normalized_data = (data - mean) / std

# Convert back to dictionary if needed
normalized_o_scores = {key: val.item() for key, val in zip(o_scores.keys(), normalized_data)}

print("Normalized Data:", normalized_o_scores)