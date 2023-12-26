import json
from KLD import kl
import tqdm



def evaluation(output_path):
    evaluation = json.load(open(output_path))
    dataset = json.load(open("../data/ESConv2.json"))
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


    for i in tqdm.tqdm(range(len(evaluation))):
        for j in range(len(evaluation[i]['dialog'])):
            if 'response' in evaluation[i]['dialog'][j]:
                strategy = find_dialog_type(evaluation[i]['dialog'][j]['id'])
                reference[strategy].append(evaluation[i]['dialog'][j]['response'])
                # find the dialog id in the dataset
                types[strategy].append(evaluation[i]['dialog'][j]['content'])

    import numpy as np
    o_score = {}
    for i in types:
        o_score[i] = np.mean(kl(reference[i], types[i]))

    return o_score

gpt3_5_path = '../gpt3.5/gpt3.5_output.json'
gpt4_path = '../gpt4/gpt4_output.json'
llama2_path = '../llama2-7b/mmistral_output.json'

gpt3_5_score = evaluation(gpt3_5_path)
gpt4_score = evaluation(gpt4_path)
llama2_score = evaluation(llama2_path)
print(gpt3_5_score)
print(gpt4_score)
print(llama2_score)


# normalizing the data
import torch

# Convert to PyTorch tensors
data_openai = torch.tensor(list(gpt3_5_score.values()), dtype=torch.float64)
data_llama2 = torch.tensor(list(llama2_score.values()), dtype=torch.float64)
data_gpt4 = torch.tensor(list(gpt4_score.values()), dtype=torch.float64)

# Concatenate tensors
combined_data = torch.cat((data_openai, data_llama2, data_gpt4), dim=0)


# Normalize combined data
def log_normalize(data):
    # Add a small constant to avoid log(0)
    # data = torch.log(data + 1e-42)  # The constant 1e-10 is arbitrary and can be adjusted
    mean = torch.mean(data)
    std = torch.std(data)
    normalized_data = (data - mean) / std
    return normalized_data

def normalize(data):
    # Normalize data
    mean = torch.mean(data)
    std = torch.std(data)
    normalized_data = (data - mean) / std
    return normalized_data


normalized_combined_data = log_normalize(combined_data)
# add the smallest value to make sure all values are positive
normalized_combined_data += torch.abs(torch.min(normalized_combined_data))
# use max data to minus all values
# normalized_combined_data = torch.max(normalized_combined_data) - normalized_combined_data
normalized_combined_data += 1

# Split and convert back to dictionaries
normalized_openai = dict(zip(gpt3_5_score.keys(), normalized_combined_data[:len(gpt3_5_score)].tolist()))
normalized_llama2 = dict(zip(llama2_score.keys(), normalized_combined_data[len(gpt3_5_score):len(gpt3_5_score) + len(llama2_score)].tolist()))
normalized_gpt4 = dict(zip(gpt4_score.keys(), normalized_combined_data[len(gpt3_5_score) + len(llama2_score):].tolist()))

print("Normalized OpenAI Data:", normalized_openai)
print("Normalized LLAMA2 Data:", normalized_llama2)
print("Normalized GPT4 Data:", normalized_gpt4)

# do a plot
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 10))
plt.bar(np.arange(len(normalized_openai)), normalized_openai.values(), width=0.25, label="OpenAI")
plt.bar(np.arange(len(normalized_gpt4))+0.25, normalized_gpt4.values(), width=0.25, label="GPT4")
plt.bar(np.arange(len(normalized_llama2))+0.5, normalized_llama2.values(), width=0.25, label="LLAMA2")
plt.xticks(np.arange(len(normalized_openai)), normalized_openai.keys(), rotation=45)
plt.ylabel("Normalized Scores")
plt.legend()
plt.show()