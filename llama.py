from llama_cpp import Llama
import json
import tqdm

llm = Llama(model_path="/Users/jesse/Doc/github/llama/chat/Llama-2-7b-chat-hf/ggml-model-q4_0.gguf",
            n_ctx=2048,
            verbose=False)


eval_dataset = json.load(open("evaluation2.json"))
# evaluation the model based on evaluation json
output = [{} for _ in range(len(eval_dataset))]

def create_completion(message):
    response = llm.create_chat_completion(
        messages=message,
        temperature=0.7,
        model="counselor",
    )
    return response

responses = []
for i in tqdm.tqdm(range(len(eval_dataset))):
    output[i]["id"] = eval_dataset[i]["id"]
    message = [
        {"role": "system",
         "content": "You are a professional mental counselor herre to support."}
    ]
    output[i]["dialog"] = [
        {
            "role": k["speaker"],
            "content": k["content"]
        }
        for k in eval_dataset[i]['dialog']
    ]
    for j in tqdm.tqdm(range(len(eval_dataset[i]["dialog"]))):
        role = eval_dataset[i]["dialog"][j]["speaker"]
        if role == "seeker":
            message.append({"role": "user", "content": eval_dataset[i]["dialog"][j]["content"]})
            # if len(eval_dataset[i]['dialog']) > j+1 and 'feedback' in eval_dataset[i]['dialog'][j+1]['annotation'].keys():

        if role == "supporter":
            message.append({"role": "assistant", "content": eval_dataset[i]["dialog"][j]["content"]})
            response = create_completion(message)
            output[i]["dialog"][j]["response"] = response['choices'][0]['message']['content']
            responses.append(response)
            # print(response['choices'][0]['message']['content'])
            json.dump(responses, open("response.json", "w"), indent=4)


    # save the output to json
    json.dump(output, open("output.json", "w"))