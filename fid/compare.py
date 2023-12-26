openai = {
    "Question": 3.2667107224410092e+40,
    "Self-disclosure": 2.535301200456459e+31,
    "Affirmation and Reassurance": 5.981525981032121e+36,
    "Providing Suggestions": 1.2379400392853803e+27,
    "Others": 8.112963841460668e+31,
    "Reflection of feelings": 1.2379400392853803e+27,
    "Information": 5.070602400912918e+30,
    "Restatement or Paraphrasing": 1.8276884942042593e+36,
}

llama2 = {
    "Question": 5.902958103587057e+20,
    "Self-disclosure": 2.9206669829258405e+33,
    "Affirmation and Reassurance": 2.5442254606820655e+35,
    "Providing Suggestions": 4.460149039706125e+43,
    "Others": 4.5556193445701994e+29,
    "Reflection of feelings": 1.0141204801825835e+31,
    "Information": 4.436777100798803e+30,
    "Restatement or Paraphrasing": 2.337230794170798e+30,
}

gpt4 = {
    "Question": 1.4462000594139885e+39,
    "Self-disclosure ": 2.839537344511234e+32,
    "Affirmation and Reassurance": 1.298074214633707e+33,
    "Providing Suggestions": 3.1691265005705735e+29,
    "Others": 8.36826229095459,
    "Reflection of feelings ": 6.084722881095501e+32,
    "Information": 8.307674973655724e+35,
    "Restatement or Paraphrasing": 1.5548526893424376e+30,
}

# normalizing the data
import torch

# Convert to PyTorch tensors
data_openai = torch.tensor(list(openai.values()), dtype=torch.float64)
data_llama2 = torch.tensor(list(llama2.values()), dtype=torch.float64)
data_gpt4 = torch.tensor(list(gpt4.values()), dtype=torch.float64)

# Concatenate tensors
combined_data = torch.cat((data_openai, data_llama2, data_gpt4), dim=0)


# Normalize combined data
def log_normalize(data):
    # Add a small constant to avoid log(0)
    data = torch.log(data + 1e-42)  # The constant 1e-10 is arbitrary and can be adjusted
    mean = torch.mean(data)
    std = torch.std(data)
    normalized_data = (data - mean) / std
    return normalized_data


normalized_combined_data = log_normalize(combined_data)
# add the smallest value to make sure all values are positive
normalized_combined_data += torch.abs(torch.min(normalized_combined_data))
# use max data to minus all values
normalized_combined_data = torch.max(normalized_combined_data) - normalized_combined_data
normalized_combined_data += 1

# Split and convert back to dictionaries
normalized_openai = dict(zip(openai.keys(), normalized_combined_data[:len(openai)].tolist()))
normalized_llama2 = dict(zip(llama2.keys(), normalized_combined_data[len(openai):len(openai) + len(llama2)].tolist()))
normalized_gpt4 = dict(zip(gpt4.keys(), normalized_combined_data[len(openai) + len(llama2):].tolist()))

print("Normalized OpenAI Data:", normalized_openai)
print("Normalized LLAMA2 Data:", normalized_llama2)
print("Normalized GPT4 Data:", normalized_gpt4)

# do a plot
import matplotlib.pyplot as plt
import numpy as np

# plt.figure(figsize=(10, 5))
plt.bar(np.arange(len(normalized_openai)), normalized_openai.values(), width=0.25, label="OpenAI")
plt.bar(np.arange(len(normalized_llama2))+0.25, normalized_llama2.values(), width=0.25, label="LLAMA2")
plt.bar(np.arange(len(normalized_gpt4))+0.5, normalized_gpt4.values(), width=0.25, label="GPT4")
plt.xticks(np.arange(len(normalized_openai)), normalized_openai.keys(), rotation=45)
plt.ylabel("Normalized Scores")
plt.legend()
plt.show()