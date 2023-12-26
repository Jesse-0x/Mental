import json

print('Loading the models...')
from KLD import kl
import tqdm

evaluation = json.load(open("../llama2-7b/mmistral_output.json"))
dataset = json.load(open("../llama2-7b/ESConv2.json"))
reference = {
    "Question": [],
    "Self-disclosure": [],
    "Affirmation and Reassurance": [],
    "Providing Suggestions": [],
    "Others": [],
    "Reflection of feelings": [],
    "Information": [],
    "Restatement or Paraphrasing": [],
}
types = {
    "Question": [],
    "Self-disclosure": [],
    "Affirmation and Reassurance": [],
    "Providing Suggestions": [],
    "Others": [],
    "Reflection of feelings": [],
    "Information": [],
    "Restatement or Paraphrasing": [],
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

# overall: -3.2882782293517914e+26
for i in types:
    print('-------------------')
    print(i, kl(reference[i], types[i]))
