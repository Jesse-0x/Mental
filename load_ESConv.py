import json

dataset = json.load(open('ESConv.json'))

new_dataset = [{} for i in range(len(dataset))]

for i in range(len(dataset)):
    new_dataset[i]['id'] = dataset[i]['id']
    new_dataset[i]['dialog'] = dataset[i]['dialog']
    for j in range(len(new_dataset[i]['dialog'])):
        new_dataset[i]['dialog'][j].pop('annotation', None)

with open('evaluation.json', 'w') as outfile:
    json.dump(new_dataset, outfile, indent=4)