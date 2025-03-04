import asyncio
import torch
from typing import List
from googlesearch import search as _search
from bs4 import BeautifulSoup
from charset_normalizer import detect
from requests_html import AsyncHTMLSession
import urllib3
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_name = 'cuda:0'
device = torch.device(device_name)

# 載入 LLM 模型
# tokenizter
tokenizer = AutoTokenizer.from_pretrained(
    "shenzhi-wang/Llama3-8B-Chinese-Chat", 
    token="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"
)
tokenizer.pad_token = tokenizer.eos_token
# quantized model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_compute_dtype=torch.float16
)
quantized_model = AutoModelForCausalLM.from_pretrained(
    "shenzhi-wang/Llama3-8B-Chinese-Chat", 
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    token="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"
)

def generate_response(_model: AutoModelForCausalLM, inputs: dict) -> str:
    output = _model.generate(
        inputs['input_ids'],
        attention_mask = inputs["attention_mask"],
        max_new_tokens=128,
        do_sample=True,
        temperature=0.1,
    )
    text_response = tokenizer.decode(output[0])
    text_response = text_response.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
    return text_response

# 搜尋工具相關程式碼
urllib3.disable_warnings()

async def worker(s: AsyncHTMLSession, url: str):
    try:
        header_response = await asyncio.wait_for(s.head(url, verify=False), timeout=10)
        if 'text/html' not in header_response.headers.get('Content-Type', ''):
            return None
        r = await asyncio.wait_for(s.get(url, verify=False), timeout=10)
        return r.text
    except:
        return None

async def get_htmls(urls):
    session = AsyncHTMLSession()
    tasks = (worker(session, url) for url in urls)
    return await asyncio.gather(*tasks)

async def search(keyword: str, n_results: int = 3) -> List[str]:
    keyword = keyword[:100]
    results = list(_search(keyword, n_results * 2, lang="zh", unique=True))
    results = await get_htmls(results)
    results = [x for x in results if x is not None]
    results = [BeautifulSoup(x, 'html.parser') for x in results]
    results = [''.join(x.get_text().split()) for x in results if detect(x.encode()).get('encoding') == 'utf-8']
    return results[:n_results]

# 向量檢索輔助函式：將文字切塊
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# 計算兩個向量的 cosine similarity
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# 定義 LLMAgent 類別
class LLMAgent():
    def __init__(self, role_description: str, task_description: str, llm: str = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"):
        self.role_description = role_description
        self.task_description = task_description
        self.llm = llm
    def inference(self, message: str) -> str:
        messages = [
            {"role": "system", "content": f"{self.role_description}"},
            {"role": "user", "content": f"{self.task_description}\n---\n{message}"},
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(device)
        return generate_response(quantized_model, inputs)

# Agent 定義
question_extraction_agent = LLMAgent(
    role_description="""
你是一名摘要專家，擅長將使用者的問題敘述簡化，去除無關的部分，只留下使用者問題的重點。
你的任務是直接輸出簡化後的問題，不需要提供任何解釋。

### 範例輸入
熊信寬，藝名熊仔，是臺灣饒舌創作歌手。2022年獲得第33屆金曲獎最佳作詞人獎，2023年獲得第34屆金曲獎最佳華語專輯獎。請問熊仔的碩班指導教授為？

### 範例輸出
熊仔的碩班指導教授為？
""",
    task_description="請幫我簡化以下問題：",
)

keyword_extraction_agent = LLMAgent(
    role_description="""
你是一名關鍵字抽取專家，擅長從使用者的問題中抽取關鍵字。
你的任務是直接輸問題的關鍵字，不需要提供任何解釋。

### 範例輸入
熊信寬，藝名熊仔，是臺灣饒舌創作歌手。2022年獲得第33屆金曲獎最佳作詞人獎，2023年獲得第34屆金曲獎最佳華語專輯獎。請問熊仔的碩班指導教授為？

### 範例輸出
熊仔、碩班、指導教授
""",
    task_description="請幫我抽取以下問題的關鍵字：",
)

qa_agent = LLMAgent(
    role_description="""
你是一名人工智慧助理，擅長根據參考資料回答使用者的問題。
""",
    task_description="請根據參考資料回答以下問題，你必須要假設參考資料都是最新的資訊，輸出一個答案。",
)

# RAG pipeline
async def pipeline(question: str) -> str:
    # 取得關鍵字與簡化後的問題
    keywords = keyword_extraction_agent.inference(question)
    keyquestion = question_extraction_agent.inference(question)
    
    # 取得搜尋結果並將結果分塊
    search_results = await search(keyword=keywords)
    chunks = []
    for result in search_results:
        chunks.extend(chunk_text(result, chunk_size=512))
    
    # 初始化向量模型並計算簡化問題的向量
    embed_model = SentenceTransformer('shibing624/text2vec-base-chinese')
    question_embedding = embed_model.encode(keyquestion)
    
    # 計算各個分塊與問題的相似度
    chunk_similarities = []
    for chunk in chunks:
        chunk_embedding = embed_model.encode(chunk)
        sim = cosine_similarity(question_embedding, chunk_embedding)
        chunk_similarities.append((sim, chunk))
    
    # 選取相似度最高的前 5 個分塊作為參考資料
    top_chunks = [chunk for sim, chunk in sorted(chunk_similarities, key=lambda x: x[0], reverse=True)[:5]]
    context = "\n- ".join(top_chunks)
    
    # 建立給 QA Agent 的提示
    rag_prompt = f"### 參考資料\n- {context}\n\n### 問題\n{keyquestion}\n"
    print(rag_prompt)
    answer = qa_agent.inference(rag_prompt)
    return answer

# 主程式：使用者可自行輸入問題
if __name__ == '__main__':
    q = input("請輸入問題：")
    answer = asyncio.run(pipeline(q))
    print("### 答案\n", answer)

# Reference：https://colab.research.google.com/drive/1OGEOSy-Acv-EwuRt3uYOvDM6wKBfSElD?usp=sharing
