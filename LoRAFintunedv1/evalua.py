from llama_cpp import llama_chat_format
import json
import tqdm

model_format = 'llama-2'

eval_dataset = json.load(open("ESConv2.json"))
eval_ids = json.load(open("evaluation_ids2.json"))
# evaluation the model based on evaluation json
output = [{} for _ in range(len(eval_dataset))]

import mlx
def create_completion(message):
    text = llama_chat_format.format_llama2(message).prompt
    model, tokenizer = load_model(args.model)
    return response

responses = []
for i in tqdm.tqdm(range(len(eval_dataset))):
    output[i]["id"] = eval_dataset[i]["id"]
    message = [
        {"role": "system",
         "content": "You are a professional mental counselor here to support."}
    ]
    output[i]["dialog"] = [
        {
            "role": k["speaker"],
            "content": k["content"],
            "id": k["id"],
        }
        for k in eval_dataset[i]['dialog']
    ]
    for j in tqdm.tqdm(range(len(eval_dataset[i]["dialog"]))):
        role = eval_dataset[i]["dialog"][j]["speaker"]
        if role == "seeker":
            message.append({"role": "user", "content": eval_dataset[i]["dialog"][j]["content"]})

        if role == "supporter":
            if eval_dataset[i]["dialog"][j]["id"] in eval_ids:
                response = create_completion(message)
                output[i]["dialog"][j]["response"] = response['choices'][0]['message']['content']
                responses.append(response)
                json.dump(responses, open("response.json", "w"), indent=4)
            message.append({"role": "assistant", "content": eval_dataset[i]["dialog"][j]["content"]})


    # save the output to json
    json.dump(output, open("mmistral_output.json", "w"), indent=4)