import json
print('Loading the models...')
from FID import fid
import tqdm
evaluation = json.load(open("../llama2-7b/mmistral_output.json"))
dataset = json.load(open("../llama2-7b/ESConv2.json"))
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

#overall: -3.2882782293517914e+26
for i in types:
    print('-------------------')
    print(i, fid(reference[i], types[i]))

# -------------------
# Question 5.902958103587057e+20
# -------------------
# Self-disclosure -2.9206669829258405e+33
# -------------------
# Affirmation and Reassurance -2.5442254606820655e+35
# -------------------
# Providing Suggestions 4.460149039706125e+43
# -------------------
# Others 4.5556193445701994e+29
# -------------------
# Reflection of feelings 1.0141204801825835e+31
# -------------------
# Information -4.436777100798803e+30
# -------------------
# Restatement or Paraphrasing 2.337230794170798e+30

# normalizing the data
from sklearn.preprocessing import MinMaxScaler
# pip install scikit-learn
import numpy as np
scaler = MinMaxScaler()
for i in reference:
    reference[i] = scaler.fit_transform(np.array(reference[i]).reshape(-1, 1))
    types[i] = scaler.fit_transform(np.array(types[i]).reshape(-1, 1))
