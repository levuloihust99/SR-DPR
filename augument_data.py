import os
import json
import pandas as pd
from tqdm import tqdm
import argparse

import aiohttp
import asyncio

async def post(session, url, headers, data):
    """_summary_

    Args:
        url (_type_): _description_
        headers (_type_): _description_
        data (_type_): _description_
    """
    # async with aiohttp.ClientSession() as session:
    async with session.post(url, headers=headers, data=data) as response:
        return await response.json()


async def send_request(question, content, opt):
    questions = [question]
    async with aiohttp.ClientSession() as session:
        # Prompt lấy n_1 câu hỏi dựa trên question
        prompt = f'''Tạo {opt.n1} câu hỏi tương đường bằng Tiếng Việt để làm tập eval cho task retrievelQA:\n
        {question}\nCâu hỏi được liệt kê theo numbering 1,2,3
        '''
        payload = json.dumps({
            "prompt": prompt,
            "backend": opt.backend
        })
        headers = {
            'Content-Type': 'application/json'
        }
        # response = requests.request("POST", opt.url, headers=headers, data=payload)
        response = await post(session, opt.url, headers=headers, data=payload)
        if response.get('completion') is not None: 
            for q in response['completion'].split('\n'):
                try:
                    questions.append(q.split('. ', 1)[1])
                except Exception as error:
                    print(error) 
                    
        # Prompt lấy n_2 câu hỏi dựa trên content
        prompt=f'''Đặt {opt.n2} câu hỏi bằng Tiếng Việt để làm tập eval cho task retrievelQA dựa trên đoạn văn sau:\n
        {content}\nCâu hỏi được liệt kê theo numbering 1,2,3
        '''
        payload = json.dumps({
            "prompt": prompt,
            "backend": opt.backend
        })
        # response = requests.request("POST", opt.url, headers=headers, data=payload)
        response = await post(session, opt.url, headers=headers, data=payload)
        if response.get('completion') is not None:
            for q in response['completion'].split('\n'):
                try:
                    questions.append(q.split('. ', 1)[1])
                except Exception as error:
                    print(error) 
        return questions


async def main(opt):
    """_summary_
    """
    # Read Data
    dfs = pd.read_excel(opt.data_file, None)
    sheet = list(dfs.keys())[0] if opt.sheet is None else opt.sheet
    df = dfs.get(sheet)
    data = [(i,j) for i,j in zip(df['Question'],df['Answer'])]
    writer = open(opt.output_file,'a+',encoding='utf8')
    sample_id = 0 
    for idx, (question, content) in tqdm(enumerate(data)):
        questions = await send_request(question, content, opt)
        item = {
                "sample_id": sample_id,
                "questions": questions,
                "positive_contexts": [
                    {
                        "title": question,
                        "text": content
                    }
                ],
                "hardneg_contexts": []
            }
        sample_id+=1
        writer.write(json.dumps(item, ensure_ascii= False)+'\n')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", "-d", required=True, type=str, help="Path to excel file")
    parser.add_argument("--sheet", "-s", required=False, default=None, help="Sheet Name")
    parser.add_argument("--output-file", "-o", required=False, default="./data/hsc_newdata/train_data.jsonl")
    parser.add_argument("--url", required=False, default="http://103.252.1.144:5777/text2text",
                        help="url to send request to text2text")
    parser.add_argument("--n1", required=False, default=3, type=int, help="Number of question which parapharse from question")
    parser.add_argument("--n2", required=False, default=3, type=int, help="Number of question which generate from content")
    parser.add_argument("--backend", required=False, default="openai", help="Number of question which generate from content")
    opt = parser.parse_args()
    asyncio.run(main(opt))