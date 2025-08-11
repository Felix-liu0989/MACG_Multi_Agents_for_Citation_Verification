import json_repair
from openai import OpenAI
import os
from dotenv import load_dotenv
import sys
sys.path.append("/home/liujian/project/2025-07/A2R-code-reproduction/src")
from citegeist.utils.llm_clients.perplexity_deep_research import PerplexityDeepResearchClient
from citegeist.utils.infer import jsonl2json
load_dotenv()
import json,jsonlines
import time
from citegeist.utils.infer import load_processed_ids
from citegeist.utils.prompts import (
    process_data_for_related_work_prompt_for_baselines
)

api_key = os.getenv("OPENROUTER_API_KEY")
client = PerplexityDeepResearchClient(api_key, "perplexity/sonar-deep-research")


with open("/home/liujian/project/2025-07/A2R-code-reproduction/datasets/arxiv_73/eval_set.json", "r") as f:
    data = json.load(f)


def process_with_checkpoint(data, output_file):
   # 加载已处理的ID
   processed_ids = load_processed_ids(output_file)
   print(f"已处理 {len(processed_ids)} 个项目，从第 {len(processed_ids)} 个开始继续...")

   # 过滤出未处理的数据
   remaining_data = []
   for id, item in enumerate(data):
      if id not in processed_ids:
         remaining_data.append(id)
   
   print(f"剩余 {len(remaining_data)} 个项目需要处理")
   for id in remaining_data:
      item = data[id]
      title = item["title"]
      if "Adapting Pretrained ViTs with Convolution Injector for Visuo-Motor Control" in title or "Enhancing Temporal Consistency in Video Editing by Reconstructing Videos with 3D Gaussian Splatting" in title or "The Price of Implicit Bias in Adversarially Robust Generalization" in title or "Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models" in title or "Eye-for-an-eye: Appearance Transfer with Semantic Correspondence in Diffusion Models" in title or "Searching Priors Makes Text-to-Video Synthesis Better" in title:
         abstract = item["abstract"]
         paper_id = item["paper_id"]
         content = f"Title: {title}\nAbstract: {abstract}"
         try:
            result = client.get_completion(process_data_for_related_work_prompt_for_baselines(content))
            print(result)
            result = result.replace("```json", "").replace("```", "")
            result = json_repair.loads(result)
            item["related_work"] = result
            with jsonlines.open(output_file, "a") as writer:
               writer.write(item)
            print(f"已完成第 {id} 项")
         except Exception as e:
            print(e)
            continue



if __name__ == "__main__":
   process_with_checkpoint(data, "/home/liujian/project/2025-07/A2R-code-reproduction/results_baselines/perplexity_deep_research_related_work_new.jsonl")
   jsonl2json("/home/liujian/project/2025-07/A2R-code-reproduction/results_baselines/perplexity_deep_research_related_work_new.jsonl",
           "/home/liujian/project/2025-07/A2R-code-reproduction/results_baselines/perplexity_deep_research_related_work_new.json")
# completion = client.chat.completions.create(
#   model="perplexity/sonar-deep-research",#"openai/gpt-4o-mini-search-preview",
#   messages=[
#     {
#       "role": "user",
#       "content": prompt.format(target_paper=target_paper)
#     }
#   ],
# )
# print(completion.choices[0].message.content)