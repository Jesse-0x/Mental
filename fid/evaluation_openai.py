import json
print('Loading the models...')
# from FID import fid
import tqdm
evaluation = json.load(open("../gpt3.5/gpt3.5_output.json"))
dataset = json.load(open("../gpt3.5/ESConv2.json"))
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
raise('wtf')
print('Start calculating FID')

#overall: -3.2882782293517914e+26
for i in types:
    print('-------------------')
    print(i, fid(reference[i], types[i]))



# normalizing the data
o_scores = {
    "Question":3.2667107224410092e+40,
    "Self-disclosure":2.535301200456459e+31,
    "Affirmation and Reassurance":-5.981525981032121e+36,
    "Providing Suggestions":1.2379400392853803e+27,
    "Others":8.112963841460668e+31,
    "Reflection of feelings":1.2379400392853803e+27,
    "Information":-5.070602400912918e+30,
    "Restatement or Paraphrasing":1.8276884942042593e+36,
}
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