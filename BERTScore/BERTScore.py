from bert_score import score
import json
import tqdm


def BERTScore(reference, generated):
    P, R, F1 = score(generated, reference, lang="en", verbose=True)
    return F1.mean().item()


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

    # order: P, R, F1
    scores = {
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
        o_score[i] = BERTScore(reference[i], types[i])

    return o_score


all_score_path = ['../gpt3.5/gpt3.5_output.json', '../gpt4/gpt4_output.json', '../llama2-7b/mmistral_output.json',
                  '../mixtral/mixtral_output.json', '../LoRAFintunedv1/LoRA_output.json']

all_score = []
for path in all_score_path:
    all_score.append(evaluation(path))
