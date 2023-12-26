from llama_cpp import Llama
import json
import tqdm

llm = Llama(model_path="/Users/jesse/Doc/github/llama/chat/Llama-2-7b-chat-hf/ggml-model-q4_0.gguf")

eval_dataset = json.load(open("evaluation.json"))
# evaluation the model based on evaluation json
output = [{} for _ in range(len(eval_dataset))]

responses = []
for i in tqdm.tqdm(range(len(eval_dataset))):
    output[i]["id"] = eval_dataset[i]["id"]
    message = [
        {"role": "system",
         "content": "You are a professional mental counselor herre to support."}
    ]
    output[i]["dialog"] = []
    for j in tqdm.tqdm(range(len(eval_dataset[i]["dialog"]))):
        role = eval_dataset[i]["dialog"][j]["role"]
        if role == "seeker":
            message.append({"role": "Client", "content": eval_dataset[i]["dialog"][j]["content"]})
            response = llm.create_chat_completion(
                messages=message,
                temperature=0.7,
                model="Counselor",
            )
            output[i]["dialog"].append([
                {"role": "Client", "content": eval_dataset[i]["dialog"][j]["content"]},
                {"role": "Counselor", "content": response['choices'][0]['message']['content']}
            ])
            responses.append(response)
            print(response['choices'][0]['message']['content'])
            json.dump(response, open("response.json", "w"))
        if role == "supporter":
            message.append({"role": "Counselor", "content": eval_dataset[i]["dialog"][j]["content"]})

    # save the output to json
    json.dump(output, open("output.json", "w"))