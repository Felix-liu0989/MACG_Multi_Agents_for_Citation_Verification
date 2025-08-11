from openai import OpenAI
import os
from dotenv import load_dotenv
import sys
sys.path.append("/home/liujian/project/2025-07/A2R-code-reproduction/src")
from citegeist.utils.llm_clients.openai_client import OpenAIClient
from citegeist.utils.llm_clients.gemini_client import GeminiClient
import json,jsonlines,json_repair
from tqdm import tqdm
load_dotenv()
from citegeist.utils.infer import load_processed_ids
from citegeist.utils.prompts import (
    process_data_for_related_work_prompt_for_baselines
)
api_key = os.getenv("OPENROUTER_API_KEY")
client = OpenAIClient(api_key, 
                      "openai/gpt-4o-mini-search-preview")
client_gemini = GeminiClient(api_key, 
                      "google/gemini-2.5-flash")
with open("/home/liujian/project/2025-07/A2R-code-reproduction/datasets/arxiv_73/eval_set.json", "r") as f:
    data = json.load(f)

def process_with_checkpoint(data, output_file):
    processed_ids = load_processed_ids(output_file)
    print(f"已处理 {len(processed_ids)} 个项目，从第 {len(processed_ids)} 个开始继续...")
    
    # 过滤出未处理的数据
    remaining_data = []
    for id,item in enumerate(data):
        if id not in processed_ids:
            remaining_data.append(id)
            
    print(f"剩余 {len(remaining_data)} 个项目")
    
    for id in tqdm(remaining_data, desc="Processing items"):
      item = data[id]
      title = item["title"]
      abstract = item["abstract"]
      paper_id = item["paper_id"]
      content = f"Title: {title}\nAbstract: {abstract}"
      try:
        result = client.get_completion(process_data_for_related_work_prompt_for_baselines(content))
        print(result)
        print("--------------------------------")
        prompt_section_1  = f"""
        You are a helpful assistant. You are given a related work with citations. 
        Please add the format '[reference number]' for citations. 
        Each citations should be in <cite></cite> tags.
        Here is the related work:
        {result}
        """
        prompt_section_2 = """
        Output must be in the following json format.
        {{
        "related_work": "...",
        "references": [
            {
            "reference_id": "1",
            "reference_info": "..."
            },
            {
            "reference_id": "2",
            "reference_info": "..."
            }
        ]
        }}
        """
        prompt = prompt_section_1 + prompt_section_2
        result = client_gemini.get_completion(prompt)
        result = result.replace("```json", "").replace("```", "")
        result = json_repair.loads(result)
        related_work = result["related_work"]
        citations = result["references"]
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
   process_with_checkpoint(data, "/home/liujian/project/2025-07/A2R-code-reproduction/results_baselines/vallina_gpt_related_work.jsonl")
            

            
            
