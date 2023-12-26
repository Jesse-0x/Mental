import nanoid
import json

# for each dialogues, assign a unique id

dataset = json.load(open('ESConv.json'))

for i in range(len(dataset)):
    for j in range(len(dataset[i]['dialog'])):
        dataset[i]['dialog'][j]['id'] = nanoid.generate()

with open('ESConv2.json', 'w') as outfile:
    json.dump(dataset, outfile, indent=4)