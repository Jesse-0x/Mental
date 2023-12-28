# here is the format:
# @register_chat_format("llama-2")
# def format_llama2(
#     messages: List[llama_types.ChatCompletionRequestMessage],
#     **kwargs: Any,
# ) -> ChatFormatterResponse:
#     _system_template = "<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>"
#     _roles = dict(user="<s>[INST]", assistant="[/INST]")
#     _messages = _map_roles(messages, _roles)
#     system_message = _get_system_message(messages)
#     if system_message:
#         system_message = _system_template.format(system_message=system_message)
#     _prompt = _format_llama2(system_message, _messages, " ", "</s>") + "[/INST]"
#     return ChatFormatterResponse(prompt=_prompt)

import json
from llama_cpp import llama_chat_format
from llama_cpp import Llama

llm = Llama(model_path="/Users/jesse/Doc/github/llama/chat/Llama-2-7b-chat-hf/ggml-model-q4_0.gguf",
            n_ctx=2048,
            verbose=True)

message = [
    {"role": "system",
     "content": "You are a professional mental counselor here to support."}
]

dataset = json.load(open("ESConv.json"))

output = []

for dialog in range(len(dataset)):
    message = [
        {"role": "system",
         "content": "You are a mental counselor here to support."}
    ]
    for i in range(len(dataset[dialog]['dialog'])):
        if dataset[dialog]['dialog'][i]['speaker'] == 'seeker':
            message.append({'role': 'user', "content": dataset[dialog]['dialog'][i]['content'].replace('\n', '')})
        elif dataset[dialog]['dialog'][i]['speaker'] == 'supporter':
            message.append({'role': 'assistant', "content": dataset[dialog]['dialog'][i]['content']})
    text = llama_chat_format.format_llama2(message).prompt
    if len(llm.tokenize(bytes(text.encode('utf-8')))) > 1500:
        print('Too long, idk what to do')
        # pass
    else:
        output.append(json.dumps({"text":text}))


# with open('train.jsonl', 'w') as f:
#     f.write('\n'.join(output))

import random
# Shuffle dataset
random.shuffle(output)

# Split dataset
train_split = int(0.7 * len(output))
valid_split = int(0.85 * len(output))

train_data = output[:train_split]
valid_data = output[train_split:valid_split]
test_data = output[valid_split:]

# Function to write data to a JSONL file
def write_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(item + '\n')

# Write to files
write_jsonl(train_data, 'train.jsonl')
write_jsonl(valid_data, 'valid.jsonl')
write_jsonl(test_data, 'test.jsonl')
