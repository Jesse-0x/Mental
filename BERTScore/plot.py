
final_data = [{'Question': 0.8566491603851318,
               'Self-disclosure': 0.8546969890594482,
               'Affirmation and Reassurance': 0.859441339969635,
               'Providing Suggestions': 0.861178457736969,
               'Others': 0.8560565114021301,
               'Reflection of feelings': 0.8560750484466553,
               'Information': 0.8566877841949463,
               'Restatement or Paraphrasing': 0.8585933446884155},
              {'Question': 0.8315087556838989,
               'Self-disclosure': 0.8251518607139587,
               'Affirmation and Reassurance': 0.829463541507721,
               'Providing Suggestions': 0.829896867275238,
               'Others': 0.8239116072654724,
               'Reflection of feelings': 0.8262116312980652,
               'Information': 0.8208602070808411,
               'Restatement or Paraphrasing': 0.8289801478385925},
              {'Question': 0.8331994414329529,
               'Self-disclosure': 0.8306849598884583,
               'Affirmation and Reassurance': 0.8407923579216003,
               'Providing Suggestions': 0.8422235250473022,
               'Others': 0.8369598388671875,
               'Reflection of feelings': 0.8383023142814636,
               'Information': 0.8348087668418884,
               'Restatement or Paraphrasing': 0.8412255048751831},
              {'Question': 0.8445861339569092,
               'Self-disclosure': 0.843146026134491,
               'Affirmation and Reassurance': 0.8448280692100525,
               'Providing Suggestions': 0.8500115275382996,
               'Others': 0.8446735739707947,
               'Reflection of feelings': 0.8435016870498657,
               'Information': 0.8360592126846313,
               'Restatement or Paraphrasing': 0.8437279462814331},
              {'Question': 0.8566401600837708,
               'Self-disclosure': 0.8433268070220947,
               'Affirmation and Reassurance': 0.8574565052986145,
               'Providing Suggestions': 0.8534427881240845,
               'Others': 0.858661949634552,
               'Reflection of feelings': 0.8468626737594604,
               'Information': 0.8431243896484375,
               'Restatement or Paraphrasing': 0.8574255704879761}]

# export in json
import json
json.dump(final_data, open("final_data.json", "w"), indent=4)

# all_score_path = ['../gpt3.5/gpt3.5_output.json', '../gpt4/gpt4_output.json', '../llama2-7b/mmistral_output.json',
#                   '../mixtral/mixtral_output.json', '../LoRAFintunedv1/LoRA_output.json']
import numpy as np
import matplotlib.pyplot as plt

# models = ['gpt3.5', 'gpt4', 'llama2', 'mixtral', 'LoRA']

data = final_data

# Extract categories and values
categories = list(data[0].keys())
values = {category: [] for category in categories}

# Populate the values for each category
for entry in data:
    for category in categories:
        values[category].append(entry[category])

from scipy.stats import zscore

combined_data = [i for i in values.values()]
combined_data = abs(zscore(combined_data))
for i in range(0, len(combined_data), 5):
    print(combined_data[i:i+5])

# Apply Z-score normalization
z_score_normalized_values = {category: abs(zscore(values[category])) for category in categories}
# z_score_normalized_values = {category: [i + abs(min(z_score_normalized_values[category]))+1 for i in z_score_normalized_values[category]] for category in categories}


# Plotting the Z-score normalized data
models = ['gpt3.5', 'gpt4', 'llama2', 'mixtral', 'Mental']
plt.figure(figsize=(12, 6))
width = 0.15  # Width of the bars
n = len(models)  # Number of models
x = np.arange(len(categories))  # Label locations

# Plot each model
for i, model in enumerate(models):
    plt.bar(x + i*width, [z_score_normalized_values[category][i] for category in categories], width, label=model)

plt.xlabel('Categories')
plt.ylabel('Z-Score Normalized Values')
plt.title('Z-Score Normalized Values per Category for Different Models')
plt.xticks(x + width*(n-1)/2, categories)
plt.legend(title="Models")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
