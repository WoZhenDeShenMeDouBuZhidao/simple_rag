# **中文說明：**

本程式實作了一個「檢索增強生成 (Retrieval Augmented Generation, RAG)」管道，利用多個 Agent 協同合作來回答使用者的問題。程式主要流程如下：

1. **模型與環境初始化：**  
   - 設定 GPU 設備並載入所需套件。  
   - 利用 Transformers 與 BitsAndBytes 以 4-bit 量化方式載入中文聊天模型（例如 "shenzhi-wang/Llama3-8B-Chinese-Chat"），並設定 tokenizer 的 pad_token 為 eos_token。

2. **生成回應函式：**  
   - `generate_response` 函式接收經過 tokenizer 處理的輸入（包含 attention_mask），然後呼叫模型的 `generate` 方法生成回應。  
   - 回應文本經過解碼與後處理後返回。

3. **網路搜尋模組：**  
   - 利用 `googlesearch`、`requests_html` 及 `BeautifulSoup` 實作非同步網頁爬取。  
   - `search` 函式根據輸入關鍵字查詢網路，並過濾出有效的網頁文字內容。

4. **向量檢索與文本分塊：**  
   - 利用 `chunk_text` 將搜尋結果分塊，便於後續計算相似度。  
   - 使用 SentenceTransformer（"shibing624/text2vec-base-chinese"）對簡化後的問題和各分塊進行向量化，再以 cosine similarity 評估相似度，選取與問題最相關的前 5 個分塊作為參考資料。

5. **多 Agent 協同：**  
   - 定義 `LLMAgent` 類別，每個 Agent 擁有不同角色與任務說明，並使用內建的 chat template 來生成提示。  
   - 包括摘要 Agent（簡化問題）、關鍵字抽取 Agent 和 QA Agent（根據參考資料生成答案）。

6. **RAG 管道 (pipeline) 執行流程：**  
   - 從使用者輸入問題開始，先用摘要與關鍵字抽取 Agent 處理問題。  
   - 接著，利用網路搜尋取得相關文本，並進行分塊與向量相似度計算，挑選出最相關的參考資料。  
   - 最後將參考資料和簡化問題組成最終提示，交由 QA Agent 生成答案並輸出。

7. **主程式：**  
   - 使用者於命令列輸入問題，程式透過 asyncio 執行管道，最終顯示生成的答案。

# **English Explanation:**

This program implements a Retrieval Augmented Generation (RAG) pipeline that uses multiple agents collaborating to answer a user's query. The overall execution flow is as follows:

1. **Model and Environment Initialization:**  
   - The GPU device is configured and required libraries are imported.  
   - A Chinese chat model (e.g., "shenzhi-wang/Llama3-8B-Chinese-Chat") is loaded using the Transformers library along with BitsAndBytes for 4-bit quantization. The tokenizer is configured such that its pad token is set to the eos token.

2. **Response Generation Function:**  
   - The `generate_response` function takes the tokenized input (which includes the attention mask) and calls the model's `generate` method to produce an output sequence.  
   - The generated token sequence is then decoded and post-processed to yield the final text response.

3. **Web Search Module:**  
   - Asynchronous web scraping is implemented using libraries like `googlesearch`, `requests_html`, and `BeautifulSoup`.  
   - The `search` function queries the web based on the provided keywords and filters valid HTML text content.

4. **Vector Retrieval and Text Chunking:**  
   - The `chunk_text` function divides the retrieved web content into manageable chunks for similarity computation.  
   - Using a SentenceTransformer model ("shibing624/text2vec-base-chinese"), the simplified query and text chunks are encoded into embeddings. Cosine similarity is calculated to select the top 5 chunks that are most relevant to the query.

5. **Multi-Agent Collaboration:**  
   - An `LLMAgent` class is defined where each agent is assigned a specific role and task description. Each agent uses a chat template to format prompts.  
   - The agents include a question simplification (summarization) agent, a keyword extraction agent, and a QA agent that generates the final answer based on the reference materials.

6. **RAG Pipeline Execution:**  
   - The pipeline begins by processing the user’s input question using the summarization and keyword extraction agents.  
   - It then performs a web search to retrieve relevant text, chunks the text, computes vector similarities, and selects the most pertinent reference chunks.  
   - These selected chunks, along with the simplified question, are assembled into a prompt which is passed to the QA agent to generate the final answer.

7. **Main Program:**  
   - The user is prompted to enter a question on the command line.  
   - The program executes the pipeline asynchronously, and the final generated answer is printed out.

# **File Structure:**

- `test.py`：Main Program  
- `1.txt`、`2.txt`：QA Examples

# **Notes:**

- Python version: 3.10.16  
- Chat model and embedding model may not perform well for non-Chinese queries.
