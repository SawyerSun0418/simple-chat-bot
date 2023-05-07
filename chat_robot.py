
import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]


def chat_with_robot(prompt, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    conversation_history.append(f"你是一个聊天机器人")
    conversation_history.append(f"用户: {prompt}")

    openai_response = openai.Completion.create(
        engine="text-davinci-002",
        prompt="\n".join(conversation_history + ["机器人:"]),
        temperature=0.7,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0.3,
        presence_penalty=0.3,
        stop=["机器人:"],
    )

    response = openai_response.choices[0].text.strip()
    conversation_history.append(f"机器人: {response}")

    return response, conversation_history


conversation_history = None 
print("和聊天机器人交谈吧！（输入 'exit' 退出）")

while True:
    user_input = input("你: ")
    if user_input.lower() == "exit":
        break

    response, conversation_history = chat_with_robot(user_input, conversation_history)
    print(f"机器人: {response}")
