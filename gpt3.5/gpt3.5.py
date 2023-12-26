import json
import tqdm
import os

eval_dataset = json.load(open("ESConv2.json"))
eval_ids = json.load(open("evaluation_ids2.json"))
# evaluation the model based on evaluation json
output = [{} for _ in range(len(eval_dataset))]

# use openai api to generate the response
from openai import OpenAI
from dotenv import load_dotenv
from openai import api_key
load_dotenv(".env")
model = 'gpt-3.5-turbo-1106'
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")
client = OpenAI()
def create_completion(message):
    response = client.chat.completions.create(
        model=model,
        messages=message,
        temperature=0.7,
    )
    return json.loads(response.model_dump_json())


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
    json.dump(output, open("gpt3.5_output.json", "w"), indent=4)
