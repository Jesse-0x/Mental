
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

dataset = json.load(open("ESConv.json"))

output = [{} for _ in range(len(dataset))]

for dialog in range(len(dataset)):
    text = ("[INST] <<SYS>>You are a mental counselor here to support. Talk to them in daily dialog format. <</SYS>> ["
            "/INST] ")
    for i in range(len(dataset[dialog]['dialog'])):
        if dataset[dialog]['dialog'][i]['speaker'] == 'seeker':
            text += "[INST] " + dataset[dialog]['dialog'][i]['content'] + " [/INST] "
        elif dataset[dialog]['dialog'][i]['speaker'] == 'supporter':
            text += dataset[dialog]['dialog'][i]['content'] + "\n"


