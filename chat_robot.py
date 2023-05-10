import os
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
import datetime
import pinecone


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']
    return vector



def gpt3_completion(prompt, engine='text-davinci-003', temp=0.0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['用户:', '聊天机器人:']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def load_conversation(results):
    result = list()
    for m in results['matches']:
        info = load_json('nexus/%s.json' % m['id'])
        result.append(info)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)
    messages = [i['message'] for i in ordered]
    return '\n'.join(messages).strip()


if __name__ == '__main__':
    convo_length = 30
    openai.api_key = 'XXX'
    pinecone.init(api_key='XXX', environment='XXX')
    pinecone.create_index("chat-bot", dimension=1536)
    index = pinecone.Index("chat-bot")
    while True:
        payload = list()
        a = input('\n\nUser: ')
        if a == "exit":
            break
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        message = a
        vector = gpt3_embedding(message)
        unique_id = str(uuid4())
        metadata = {'speaker': 'User', 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': unique_id}
        save_json('nexus/%s.json' % unique_id, metadata)
        payload.append((unique_id, vector))
        results = index.query(vector=vector, top_k=convo_length)
        conversation = load_conversation(results)  
        prompt = open_file('prompt_response.txt').replace('<<CONVERSATION>>', conversation).replace('<<MESSAGE>>', a)
        output = gpt3_completion(prompt)
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        message = output
        vector = gpt3_embedding(message)
        unique_id = str(uuid4())
        metadata = {'speaker': 'chat-robot', 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': unique_id}
        save_json('nexus/%s.json' % unique_id, metadata)
        payload.append((unique_id, vector))
        index.upsert(payload)
        print('\n\nchat-robot: %s' % output) 
    pinecone.delete_index("chat-bot")

    for filename in os.listdir('nexus'):
        if not filename.startswith('.'):
            file_path = os.path.join('nexus', filename)
            try:
                os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")