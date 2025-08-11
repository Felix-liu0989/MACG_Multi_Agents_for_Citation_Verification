import sys
sys.path.append("/home/liujian/project/2025-07/A2R-code-reproduction/src")
from post_hoc_rag.retriever import ArxivRetriever
from citegeist.utils.infer import load_processed_ids
from post_hoc_rag.reranker import Reranker
from post_hoc_rag.read_dataset import Reader
from citegeist.utils.llm_clients.gemini_client import GeminiClient
from citegeist.utils.llm_clients.deepseek_client import DeepSeekClient
from citegeist.utils.llm_clients.openai_client import OpenAIClient
from tqdm import tqdm
import os
from dotenv import load_dotenv
from citegeist.utils.prompts import (
    process_data_for_related_work_prompt_for_naive_rag_gpt
)
import json,jsonlines,json_repair
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
reader = Reader("/home/liujian/project/2025-07/A2R-code-reproduction/datasets/retrieval_data.json")
arxiv_corpus = reader.get_data()    

retriever = ArxivRetriever(
    model_name="/home/liujian/project/2025-07/A2R-code-reproduction/bge-m3",
    arxiv_corpus=arxiv_corpus,
    device="cuda",
    language="en",
    faiss_index_path="/home/liujian/project/2025-07/A2R-code-reproduction/faiss_index_naive_rag"
)

reranker = Reranker(
    rerank_model_name_or_path="/home/liujian/project/2025-07/A2R-code-reproduction/bge-reranker-large",
    device="cuda",
)


client_gemini = GeminiClient(api_key, 
                "google/gemini-2.5-flash")
client_deepseek = DeepSeekClient(api_key=os.getenv("DEEPSEEK_API_KEY"),
                model_name="deepseek-chat")
client_openai = OpenAIClient(api_key, 
                "openai/gpt-4o-mini-search-preview")

def process_with_checkpoint_naive_rag_gpt(data, output_file):
    processed_ids = load_processed_ids(output_file)
    print(f"已处理 {len(processed_ids)} 个项目，从第 {len(processed_ids)} 个开始继续...")
    
    # 过滤出未处理的数据
    remaining_data = []
    for id,item in enumerate(data):
        if id not in processed_ids:
            remaining_data.append(id)
            
    for id in tqdm(remaining_data, desc="Processing items"):
        item = data[id]
        title = item["title"]
        abstract = item["abstract"]
        paper_id = item["paper_id"]
        content = f"Title: {title}\nAbstract: {abstract}"
        try:
            retrieved_results = retriever.vector_retrieval(abstract,top_k=30)
            retrieved_results_list = []
            for retrieved_result in retrieved_results:
                retrieved_abstract = retrieved_result["instruction"]
                print(f"retrieved_abstract: {retrieved_abstract}")
                retrieved_reference = retrieved_result["reference"]
                retrieved_item = {
                    "instruction":retrieved_abstract,
                    "reference":retrieved_reference
                }
                retrieved_results_list.append(retrieved_item)
            reference_results = []
            for retrieved_result in retrieved_results_list:
                reference_results.append(retrieved_result['reference'])
                
            item["retrieved_results"] = retrieved_results_list
            
            reranked_results = reranker.rerank(retrieved_results_list,content,k=20)
            reranked_results_list = []
            for id,result in enumerate(reranked_results):
                title = result['instruction']
                title = title.split("\n")[0]
                print(f"title: {title}")
                abstract = result["instruction"]
                print(f"abstract: {abstract}")
                print(f"reference: {result['reference']}")
                print(f"score: {result['score']}")
                print("-"*100)
                reranked_item = {
                    "id":id+1,
                    "title":title,
                    "abstract":abstract,
                    "reference":result['reference'],
                    "score":result['score']
                }
                reranked_results_list.append(reranked_item)
            item["reranked_results"] = reranked_results_list
            prompt = process_data_for_related_work_prompt_for_naive_rag_gpt(content,reranked_results_list)
            print(prompt)
            result = client_gemini.get_completion(prompt)
            result = result.replace("```json", "").replace("```", "")
            result = json_repair.loads(result)
            related_work = result["related_work"]
            citations = result["cite_ids"]
            print(related_work)
            print(citations)
            item["related_work"] = related_work
            item["citations"] = citations
            with jsonlines.open(output_file, "a") as writer:
                writer.write(item)
                print(f"已完成第 {id} 项")
        except Exception as e:
            print(e)
            continue

if __name__ == "__main__":
    with open("/home/liujian/project/2025-07/A2R-code-reproduction/datasets/arxiv_73/eval_set.json", "r") as f:
        data = json.load(f)
    process_with_checkpoint_naive_rag_gpt(data, "/home/liujian/project/2025-07/A2R-code-reproduction/results_baselines/naive_rag_gemini_related_work_07_20.jsonl")