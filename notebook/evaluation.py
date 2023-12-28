import json
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

dataset = json.load(open("../data/ESConv2.json"))

def find_dialog_type(dialog_id):
    for i in range(len(dataset)):
        for j in range(len(dataset[i]['dialog'])):
            if dataset[i]['dialog'][j]['id'] == dialog_id:
                return dataset[i]['dialog'][j]['annotation']['strategy']
    raise Exception('IDK What the hack is happening here')

def find_type_dialog(dialog_type):
    dialog_ids = []
    for i in range(len(dataset)):
        for j in range(len(dataset[i]['dialog'])):
            if 'strategy' in dataset[i]['dialog'][j]['annotation'].keys() and dataset[i]['dialog'][j]['annotation']['strategy'] == dialog_type:
                dialog_ids.append(dataset[i]['dialog'][j]['id'])
    return dialog_ids

def get_dialog_by_id(dialog_id):
    for i in range(len(dataset)):
        for j in range(len(dataset[i]['dialog'])):
            if dataset[i]['dialog'][j]['id'] == dialog_id:
                return dataset[i]['dialog'][j]['content']
    raise Exception('IDK What the hack is happening here')

# find all the dialog sentence for all different strategy
for i in tqdm.tqdm(range(len(types))):
    for dialog_id in find_type_dialog(list(types.keys())[i]):
        types[list(types.keys())[i]].append(get_dialog_by_id(dialog_id))

# train a model that can identify the type of problem
# using Transformer, input use tokenizer and then output with 8 different categories
import transformers

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')


class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.linear(output[1])
        output = self.softmax(output)
        return output

    def model_parameter_size(model):
        size = sum(p.numel() for p in model.parameters() if p.requires_grad)

model = BertClassifier(8)
model.to('mps')
new_types = {key: [] for key in types.keys()}
for i in range(len(types)):
    for j in range(len(types[list(types.keys())[i]])):
        tag = tokenizer(types[list(types.keys())[i]][j], return_tensors='pt', padding=True, pad_to_multiple_of=512)
        if len(tag) > 512: print('wtf')
        else: new_types[list(types.keys())[i]].append(tag)

import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data_dict):
        self.samples = []
        self.labels = []

        # Assign a unique integer to each key for classification
        self.label_dict = {label: idx for idx, label in enumerate(data_dict.keys())}

        for label, items in data_dict.items():
            for item in items:
                self.samples.append((item['input_ids'], item['attention_mask']))
                self.labels.append(self.label_dict[label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, attention_mask = self.samples[idx]
        label = self.labels[idx]
        return input_ids.squeeze(0), attention_mask.squeeze(0), label

# Initialize the dataset
from sklearn.model_selection import train_test_split

def split_data(data_dict, test_size=0.2):
    train_data = {key: [] for key in data_dict.keys()}
    val_data = {key: [] for key in data_dict.keys()}

    for key, items in data_dict.items():
        train_items, val_items = train_test_split(items, test_size=test_size)
        train_data[key] = train_items
        val_data[key] = val_items

    return train_data, val_data

# Split the data
train_data, val_data = split_data(new_types)

# Initialize the datasets
train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)

# Initialize the DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)  # Shuffle is usually not needed for validation/test sets

from transformers import BertForSequenceClassification, AdamW

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Assuming you're using a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


from tqdm import tqdm
import torch.nn.functional as F


def train_model(model, train_loader, val_loader, optimizer, epochs=4):
    for epoch in range(epochs):
        model.train()
        total_loss, total_accuracy = 0, 0

        # Training loop
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            model.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            # calculate the accuracy
            logits = outputs.logits
            predictions = torch.argmax(F.softmax(logits, dim=1), dim=1)
            print('accuracy', (predictions == labels).sum().item() / len(labels), end=' ')
            print('loss', loss.item())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)

        # Validation loop
        model.eval()
        total_eval_accuracy = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Evaluating Epoch {epoch+1}"):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits

                predictions = torch.argmax(F.softmax(logits, dim=1), dim=1)
                total_eval_accuracy += (predictions == labels).sum().item()

        avg_val_accuracy = total_eval_accuracy / len(val_dataset)
        # save
        model.save_pretrained('model')

        print(f"Epoch {epoch+1}:")
        print(f"  Average training loss: {avg_train_loss:.4f}")
        print(f"  Validation Accuracy: {avg_val_accuracy:.4f}")


# load the model
model = BertForSequenceClassification.from_pretrained('modela', num_labels=8)
# Training the model
train_model(model, train_loader, val_loader, optimizer)
