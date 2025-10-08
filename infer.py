from citegeist.generator import Generator
import os
import json,jsonlines
import time
from citegeist.utils.infer import load_processed_ids
import traceback
import time


generator = Generator(
   llm_provider="gemini",  # Choice of: "azure" (OpenAI Studio), "anthropic", "gemini", "mistral", and "openai"
   api_key=os.environ.get("OPENROUTER_API_KEY"), # Here, you will need to set the respective API key
   model_name="deepseek/deepseek-v3.2-exp", # Choose the model that the provider supports
   database_uri= os.environ.get("MILVUS_URI", ""),# os.environ.get("MILVUS_URI", ""),  # Set the path (local) / url (remote) for the Milvus DB connection
   database_token= os.environ.get("MILVUS_TOKEN", ""),  # Optionally, also set the access token (you DON'T need to set this when using the locally hosted Milvus Database)
)

def jsonl2json(jsonl_file, json_file):
   with jsonlines.open(jsonl_file, "r") as reader:
      data = [line for line in reader]
   with open(json_file, "w", encoding="utf-8") as f:
      json.dump(data, f, ensure_ascii=False, indent=4)


def process_single_item(content,arxiv_id):
   result,token_usage = generator.generate_related_work_MACG(content, 20, 2, 0.1,arxiv_id)
   
   return result,token_usage


def process_with_checkpoint(data, output_file):
   # 加载已处理的ID
   processed_ids = load_processed_ids(output_file)
   print(f"已处理 {len(processed_ids)} 个项目，从第 {len(processed_ids)} 个开始继续...")

   # 过滤出未处理的数据
   remaining_data = []
   for id, item in enumerate(data):
      if id not in processed_ids:
         remaining_data.append(id)
   
   all_time = 0
   dag_time = 0
   related_work_time = 0
   feedback_time = 0
   summarization_time = 0
   print(f"剩余 {len(remaining_data)} 个项目需要处理")
   for id in remaining_data[:2]:
      item = data[id]
      title = item["title"]
      arxiv_id = item["paper_id"].replace(".", "_")
      abstract = item["abstract"]
      content = f"Title: {title}\nAbstract: {abstract}"
      try:
         result,token_usage = generator.generate_related_work_MACG(content, 10, 2, 0.25, arxiv_id)
         print(token_usage)
         # selected_papers = result["selected_papers"]
         time_dict = result["time_dict"]
         all_time = time_dict["all_time"]
         dag_time = time_dict["dag_time"]
         related_work_time = time_dict["related_work_time"]
         feedback_time = time_dict["feedback_time"]
         summarization_time = time_dict["summarization_time"]
         all_time += all_time
         dag_time += dag_time
         related_work_time += related_work_time
         feedback_time += feedback_time
         summarization_time += summarization_time
         # with open(f"results/{arxiv_id}.json", "w", encoding="utf-8") as f:
         #    json.dump(selected_papers, f, ensure_ascii=False, indent=4)
         # ablation_experiment_without_summarization(content, 20, 3, 0.0)
         # generator.ablation_experiment_without_feedback_revision(content, 20, 3, 0.0)
         item["related_work"] = result
         with jsonlines.open(output_file, "a") as writer:
            writer.write(item)
         print(f"已完成第 {id} 项")
      except Exception as e:
         traceback.print_exc()
         continue
      
   num = len(remaining_data[:2])
   print("Time taken for all process:", round(all_time/num, 2), "seconds")
   print("Time taken for DAG:", round(dag_time/num, 2), "seconds")
   print("Time taken for related work:", round(related_work_time/num, 2), "seconds")
   print("Time taken for feedback:", round(feedback_time/num, 2), "seconds")
   print("Time taken for summarization:", round(summarization_time/num, 2), "seconds")


def process_with_checkpoint_citegeist(data, output_file):
   # 加载已处理的ID
   processed_ids = load_processed_ids(output_file)
   print(f"已处理 {len(processed_ids)} 个项目，从第 {len(processed_ids)} 个开始继续...")

   # 过滤出未处理的数据
   remaining_data = []
   for id, item in enumerate(data):
      if id not in processed_ids:
         remaining_data.append(id)
   

   print(f"剩余 {len(remaining_data)} 个项目需要处理")
   for id in remaining_data[:1]:
      item = data[id]
      title = item["title"]
      arxiv_id = item["paper_id"].replace(".", "_")
      abstract = item["abstract"]
      content = f"Title: {title}\nAbstract: {abstract}"
      try:
         result = generator.generate_related_work(content, 20, 2, 1, arxiv_id)
         selected_papers = result["selected_papers"]
         
         item["related_work"] = result
         with jsonlines.open(output_file, "a") as writer:
            writer.write(item)
         print(f"已完成第 {id} 项")
      except Exception as e:
         traceback.print_exc()
         continue

def process_with_checkpoint_only_retrieval(data):
   # 加载已处理的ID
   # processed_ids = load_processed_ids(output_file)
   # print(f"已处理 {len(processed_ids)} 个项目，从第 {len(processed_ids)} 个开始继续...")

   # 过滤出未处理的数据
   remaining_data = data
   # for id, item in enumerate(data):
   #    if id not in processed_ids:
   #       remaining_data.append(id)
   
   for item in remaining_data[:5]:
      title = item["title"]
      arxiv_id = item["paper_id"].replace(".", "_")
      abstract = item["abstract"]
      content = f"Title: {title}\nAbstract: {abstract}"
   
      try:
         generator.generate_related_work_MACG_Only_Retrieval(content, 20, 2, 0.25, arxiv_id)
            
      except Exception as e:
         traceback.print_exc()
         continue
      

def process_with_checkpoint_fast_macg(data,selected_papers_list,output_file):
   processed_ids = load_processed_ids(output_file)
   remaining_data = []
   print(f"已处理 {len(processed_ids)} 个项目，从第 {len(processed_ids)} 个开始继续...")
   
   # 过滤出未处理的数据
   for id, item in enumerate(data):
      if id not in processed_ids:
         remaining_data.append(id)
   
   print(f"剩余 {len(remaining_data)} 个项目需要处理")
   
   for id in remaining_data[:5]:
      item = data[id]
      title = item["title"]
      arxiv_id = item["paper_id"].replace(".", "_")
      abstract = item["abstract"]
      print(abstract)
      content = f"Title: {title}\nAbstract: {abstract}"
      selected_papers = selected_papers_list[id]
      try:
         result = generator.generate_related_work_MACG_FAST(content, 20, 2, 0, arxiv_id, selected_papers)
         item["related_work"] = result
         with jsonlines.open(output_file, "a") as writer:
            writer.write(item)
      except Exception as e:
         traceback.print_exc()
         continue
# path = "eval_set_50.json"
# with open(path, "r", encoding="utf-8") as f:
#    data = json.load(f)
# dir = "results"
# os.makedirs(dir, exist_ok=True)
# output = os.path.join(dir, "result_MACG_test.jsonl")

# process_with_checkpoint(data, output)

# jsonl2json(output, os.path.join(dir, "result_MACG_test.json"))

def main_fast_macg():
   path = "eval_set_50.json"
   selected_papers_path = r"D:\Mydesktop\CitAgent\Code\Supplementary material and code\selected_papers\20_2_0.json"
   with open(selected_papers_path, "r", encoding="utf-8") as f:
      selected_papers_list = json.load(f)
   with open(path, "r", encoding="utf-8") as f:
      data = json.load(f)
   dir = r"D:\Mydesktop\CitAgent\Code\Supplementary material and code\Multi_Agents_for_Citation_Verification\results"
   os.makedirs(dir, exist_ok=True)
   output_file = os.path.join(dir, "result_MACG_fast_20_2_0_1006.jsonl")
   
   process_with_checkpoint_fast_macg(data, selected_papers_list, output_file)
   jsonl2json(output_file, os.path.join(dir, "result_MACG_fast_20_2_0_1006.json"))
   
def main_macg():
   path = "eval_set_50.json"
   with open(path, "r", encoding="utf-8") as f:
      data = json.load(f)
   dir = r"D:\Mydesktop\CitAgent\Code\Supplementary material and code\Multi_Agents_for_Citation_Verification\results"
   os.makedirs(dir, exist_ok=True)
   output_file = os.path.join(dir, "result_MACG_test_1006.jsonl")
   final,token_usage = process_with_checkpoint(data, output_file)
   print(token_usage)

   jsonl2json(output_file, os.path.join(dir, "result_MACG_test_1006.json"))
   
def main_citegeist():
   path = "eval_set_50.json"
   with open(path,"r",encoding="utf-8") as f:
      data = json.load(f)
      dir = r"D:\Mydesktop\CitAgent\Code\Supplementary material and code\Multi_Agents_for_Citation_Verification\results"
   os.makedirs(dir, exist_ok=True)
   output_file = os.path.join(dir, "result_citegeist_test_diversity_1.jsonl")
   process_with_checkpoint_citegeist(data, output_file)
   jsonl2json(output_file, os.path.join(dir, "result_citegeist_test_diversity_1.json"))
   
def main_single_item():
   abstract = r"""Multi-Agent Framework for Thematically Structuring and Generation of Related Work: AI-driven survey generation has advanced rapidly, yet related work generation (RWG) remains underexplored. Unlike surveys that provide broad overviews, RWG must synthesize literature for a single focal paper, requiring contextual fit, comparison, and accurate attribution. Automated synthesis of related work sections remains challenging due to demands for coherent organization, deep analysis, and factual grounding. We propose MACG, a fully automated multi-agent framework that transforms a title and abstract into a polished related work section. MACG first performs diversity-aware semantic retrieval to select high-quality candidate papers, then leverages four specialized agents (Summarization, Organization, Integration, Fact Checking) for DAG-based taxonomy construction, feedback refinement, and dual-model verification. Evaluated on the benchmark dataset from ScholarCopilot (encompassing recent arXiv papers with verified citations), our proposed MACG method significantly outperforms the strongest existing baseline, delivering over 12\% improvement in overall text quality, more than 50\% gain in structural coherence, and about 10\% boost in citation accuracy. Ablation studies confirm the critical contributions of each agent to motivational clarity, structural coherence, and attribution precision. These results validate MACG as a new state-of-the-art for automated related work generation, offering significant enhancements in logical flow, analytical depth, and academic integrity."""
   arxiv_id = "2509_09125"
   result,token_usage = process_single_item(abstract, arxiv_id)
   print(token_usage)
   print(result)

if __name__ == "__main__":
   # main_macg()
   # main_citegeist()
   # main_fast_macg()
   main_single_item()





# # "root:Milvus"
# # "D:\Mydesktop\CitAgent\Code\Supplementary material and code\Multi_Agents_for_Citation_Verification\milvus_demo.db"
# abstract = '''
# Flow of Reasoning -- Training LLMs for Divergent Problem Solving with Minimal Examples: The ability to generate diverse solutions to a given problem is a hallmark of human creativity. This divergent reasoning is also crucial for machines, enhancing their robustness and enabling them to assist humans in many applications such as scientific discovery. However, existing approaches to multi-step reasoning with large language models (LLMs) have mostly focused only on reasoning accuracy, without further discovering more diverse valid solutions. For example, supervised fine-tuning can improve LLM reasoning quality, but requires extensive supervised data to capture the full range of possible solutions. Reinforcement learning aims to find limited highest-reward solutions while neglecting the solution diversity. To fill this gap, we propose Flow of Reasoning (FoR), an efficient diversity-seeking LLM finetuning method aimed at improving reasoning quality and diversity with minimal data. FoR formulates multi-step LLM reasoning as a Markovian flow on a DAG-structured reasoning graph. This formulation allows us to incorporate and adapt principled GFlowNet approaches, for finetuning LLMs to sample diverse reasoning paths with probabilities proportional to the (unnormalized) reward of target problems. Extensive experiments show that, with limited training examples (e.g., 15 examples), FoR enables the discovery of diverse, creative, high-quality solutions, greatly outperforming a wide range of existing inference and training methods across five challenging puzzle-solving tasks, including BlocksWorld (embodied reasoning), Game24 (math puzzle solving), Rubik's Cube (spatial reasoning), 1D-ARC (abstraction reasoning), and PrOntoQA (logical reasoning). Code is available at https://github.com/Yu-Fangxu/FoR.
# '''
# result = generator.generate_related_work_MACG(abstract, 25, 2, 0.25)
# related_work = result["related_works"]
# citations = result["citations"]
# print("related_work:")
# print(related_work)
# print("citations:")
# print(citations)
# end_time = time.time()
# print(f"Time taken: {round(end_time - start_time, 2)} seconds")
# citations = "\n".join(citations)
# result = related_work + "\n" + citations

# with open("related_work.txt", "w") as f:
#    f.write(result)