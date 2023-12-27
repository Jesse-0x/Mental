import json
from FID import fid
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
        o_score[i] = np.mean(fid(reference[i], types[i]))

    return o_score

gpt3_5_path = '../gpt3.5/gpt3.5_output.json'
gpt4_path = '../gpt4/gpt4_output.json'
llama2_path = '../llama2-7b/mmistral_output.json'
mixtral_path = '../mixtral/mixtral_output.json'

gpt3_5_score = evaluation(gpt3_5_path)
gpt4_score = evaluation(gpt4_path)
llama2_score = evaluation(llama2_path)
mixtral_score = evaluation(mixtral_path)
print(gpt3_5_score)
print(gpt4_score)
print(llama2_score)
print(mixtral_score)


# normalizing the data
import torch

# Convert to PyTorch tensors
data_openai = torch.tensor(list(gpt3_5_score.values()), dtype=torch.float64)
data_llama2 = torch.tensor(list(llama2_score.values()), dtype=torch.float64)
data_gpt4 = torch.tensor(list(gpt4_score.values()), dtype=torch.float64)
data_mixtral = torch.tensor(list(mixtral_score.values()),dtype=torch.float64)

# Concatenate tensors
combined_data = torch.cat((data_openai, data_llama2, data_gpt4, data_mixtral), dim=0)
# {'Question': 3.2667107224410092e+40, 'Self-disclosure': 2.535301200456459e+31, 'Affirmation and Reassurance': -5.981525981032121e+36, 'Providing Suggestions': 1.2379400392853803e+27, 'Others': 8.112963841460668e+31, 'Reflection of feelings': 1.2379400392853803e+27, 'Information': -5.070602400912918e+30, 'Restatement or Paraphrasing': 1.8276884942042593e+36}
# {'Question': 1.4462000594139885e+39, 'Self-disclosure': -2.839537344511234e+32, 'Affirmation and Reassurance': 1.298074214633707e+33, 'Providing Suggestions': 3.1691265005705735e+29, 'Others': 8.36826229095459, 'Reflection of feelings': -6.084722881095501e+32, 'Information': 8.307674973655724e+35, 'Restatement or Paraphrasing': 1.5548526893424376e+30}
# {'Question': 5.902958103587057e+20, 'Self-disclosure': -2.9206669829258405e+33, 'Affirmation and Reassurance': -2.5442254606820655e+35, 'Providing Suggestions': 4.460149039706125e+43, 'Others': 4.5556193445701994e+29, 'Reflection of feelings': 1.0141204801825835e+31, 'Information': -4.436777100798803e+30, 'Restatement or Paraphrasing': 2.337230794170798e+30}
# {'Question': 2.321137573660088e+26, 'Self-disclosure': -3.448009632620784e+32, 'Affirmation and Reassurance': -8.715097876569077e+29, 'Providing Suggestions': -3.1691265005705735e+29, 'Others': 1.2169445762191002e+32, 'Reflection of feelings': -3.802951800684688e+30, 'Information': 4.460149039706125e+43, 'Restatement or Paraphrasing': -2.9076862407795035e+35}

# Normalize combined data
def log_normalize(data):
    # Add a small constant to avoid log(0)
    data = torch.log(data + 1e-42)  # The constant 1e-10 is arbitrary and can be adjusted
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


normalized_combined_data = normalize(combined_data)
# add the smallest value to make sure all values are positive
normalized_combined_data += torch.abs(torch.min(normalized_combined_data))
# use max data to minus all values
# normalized_combined_data = torch.max(normalized_combined_data) - normalized_combined_data
normalized_combined_data += 1

# Split and convert back to dictionaries
normalized_openai = dict(zip(gpt3_5_score.keys(), normalized_combined_data[:len(gpt3_5_score)].tolist()))
normalized_llama2 = dict(zip(llama2_score.keys(), normalized_combined_data[len(gpt3_5_score):len(gpt3_5_score) + len(llama2_score) + len(mixtral_score)].tolist()))
normalized_gpt4   = dict(zip(gpt4_score.keys(),   normalized_combined_data[len(gpt3_5_score)+len(llama2_score):len(gpt3_5_score)+len(llama2_score)+len(gpt4_score)].tolist()))
normalized_mixtral=dict(zip(mixtral_score.keys(), normalized_combined_data[len(gpt3_5_score)+len(llama2_score)+len(gpt4_score):].tolist()))


print("Normalized OpenAI Data:", normalized_openai)
print("Normalized LLAMA2 Data:", normalized_llama2)
print("Normalized GPT4 Data:", normalized_gpt4)
print("Normalized Mixtral Data:", normalized_mixtral)





# do a plot
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 10))
plt.bar(np.arange(len(normalized_openai)), normalized_openai.values(), width=0.2, label="OpenAI")
plt.bar(np.arange(len(normalized_gpt4))+0.2, normalized_gpt4.values(), width=0.2, label="GPT4")
plt.bar(np.arange(len(normalized_llama2))+0.4, normalized_llama2.values(), width=0.2, label="LLAMA2")
plt.bar(np.arange(len(normalized_mixtral))+0.6, normalized_mixtral.values(), width=0.2, label="Mixtral")
plt.xticks(np.arange(len(normalized_openai)), normalized_openai.keys(), rotation=45)
plt.ylabel("Normalized Scores")
plt.legend()
plt.show()
