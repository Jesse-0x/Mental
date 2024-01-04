import openai

# using prompt, lstm, and set a goal

# lstm is the model to identify what should the next step to confort the user
# prompt is the model to generate the response, which will provide the emotion of the user
# and another model to identify the issue of the user
# goal is to talking therapy, which is to help the user to find the problem and solve it

import torch
import torch.nn as nn
import tqdm
import json


# Actually parallel model for identification of problem, strategy choose, and then determine the goal.
# the goal can be established and changed / filtered out by the lstm model
# strategy choose might need based on a llm / a better sentence classifier.
# Where requires researches about how prompt can reduce humiliation
# Which should be... I mean P(w|wi-1)
# more prompt meaning eliminate other probability and increase the prob of giving positive outcome.

class NextStep(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NextStep, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x


import transformers

class Goal(nn.Module):
    def __init__(self, model_name, num_labels):
        super(Goal, self).__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x
