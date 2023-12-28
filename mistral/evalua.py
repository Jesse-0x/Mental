from llama_cpp import Llama
import json
import tqdm


eval_dataset = json.load(open("ESConv2.json"))
eval_ids = json.load(open("evaluation_ids2.json"))
# evaluation the model based on evaluation json
output = [{} for _ in range(len(eval_dataset))]

llm = Llama(model_path="/Users/jesse/Doc/github/llama/chat/Llama-2-7b-chat-hf/ggml-model-q4_0.gguf",
            n_ctx=2048,
            verbose=True)
def create_completion(_message):
    text = llama_chat_format.format_llama2(_message).prompt
    rt = chat_completion(text)
    return rt


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
            message.append({"role": "user", "content": eval_dataset[i]["dialog"][j]["content"].replace('\\n', '')})

        if role == "supporter":
            if eval_dataset[i]["dialog"][j]["id"] in eval_ids:
                response = create_completion(message).replace('</s>', '')
                output[i]["dialog"][j]["response"] = response
                responses.append(response)
                json.dump(responses, open("response.json", "w"), indent=4)
            message.append({"role": "assistant", "content": eval_dataset[i]["dialog"][j]["content"]})

    # save the output to json
    json.dump(output, open("LoRA_output.json", "w"), indent=4)
