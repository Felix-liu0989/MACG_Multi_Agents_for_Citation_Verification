import json
import os
import json_repair
import jsonlines
import re
from evaluation.agents.judge import Judge
from citegeist.utils.infer import jsonl2json
from citegeist.utils.llm_clients.deepseek_client import DeepSeekClient
from citegeist.utils.llm_clients.gemini_client import GeminiClient
from eval_test import count_sentences,load_processed_ids
# "/home/liujian/project/2025-07/A2R-code-reproduction/results_baselines/citegeist/result_citegeist_extract_cited_sentences.json"
# with open("/home/liujian/project/2025-07/A2R-code-reproduction/results_our_workflow/result_our_workflow.json","r") as file:
#     data = json.load(file)

client = DeepSeekClient(
    api_key = os.environ.get("DEEPSEEK_API_KEY", ""),
    model_name = "deepseek-chat"
)
# client = GeminiClient(
#     api_key = os.environ.get("GEMINI_API_KEY", ""),
#     model_name = "gemini-2.5-flash"
# )
input_file = "/home/liujian/project/2025-07/A2R-code-reproduction/results_baselines/vallina_gpt_related_work.json"
output_file = "/home/liujian/project/2025-07/A2R-code-reproduction/results_baselines/vallina_gpt_related_work_cited_sentences.jsonl"



def process_with_checkpoint_extract_cited_sentences(input_file, output_file):
   # 加载已处理的ID
    processed_ids = load_processed_ids(output_file)
    print(f"已处理 {len(processed_ids)} 个项目，从第 {len(processed_ids)} 个开始继续...")

    # 过滤出未处理的数据
    remaining_data = []
    with open(input_file,"r",encoding="utf-8") as f:
        data = json.load(f)
    for id, item in enumerate(data):
      if id not in processed_ids:
         remaining_data.append(id)
    

    print(f"剩余 {len(remaining_data)} 个项目需要处理")
    for id in remaining_data[:]:
        item = data[id]
        if "related_work" not in item:
            continue
        related_work = item["related_work"]
        try:
            prompt = f"""
            The following is a related work section. Please extract the sentences with citations (e.g. (Mildenhall et al., 2021)) based on the information in the related work.
            For example:
            The neural radiance field (NeRF) <cite>(Mildenhall et al., 2021)</cite> is one of the advanced methodologies that
            represent scenes using implicit neural rendering with MLP.

            Here is the related work:
            {related_work}
            Please output the whole sentences with citations, not just the citations.
            Please output the result in the following List format:
            [
                "The neural radiance field (NeRF) <cite>(Mildenhall et al., 2021)</cite> is one of the advanced methodologies that
                represent scenes using implicit neural rendering with MLP.",
                ...
            ]
            """
            result = client.get_completion(prompt)
            result = result.replace("```json", "").replace("```", "")
            result = json_repair.loads(result)
            for i,quote in enumerate(result,1):
                print(f"{i}. {quote}")
                print("-"*100)
            item["quotes"] = result
            with jsonlines.open(output_file, "a") as writer:
                writer.write(item)
            print(f"已完成第 {id} 项")
        except Exception as e:
            print(e)
            continue
        
def eval_citation_baselines(input_file,output_file):
    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)
        
    judge_gemini = Judge(model="google/gemini-2.5-flash")
    judge_deepseek = Judge(model="deepseek-chat")
    
    processed_ids = load_processed_ids(output_file)
    print(f"已处理 {len(processed_ids)} 个项目，从第 {len(processed_ids)} 个开始继续...")
    
    remaining_data = []
    for id,item in enumerate(data):
      if id not in processed_ids:
         remaining_data.append(id)
    
    print(f"剩余 {len(remaining_data)} 个项目需要处理")
    
    claim_precision_list = []
    citation_precision_list = []
    reference_precision_list = []
    citation_density_list = []
    avg_citation_per_sentence_list = []
    
    for idx in remaining_data[:]:
        item = data[idx]
        related_work = item["related_work"]
        quotes = item["quotes"]
        citations = item["citations"]
        citations = [
                citation for citation in citations 
                if citation.get("paper_id") and citation["paper_id"] not in [None, "None", "null", ""]
            ]
        selected_papers = item["reranked_results"]
        ids = [i for i in citations if "paper_id" in i and i["paper_id"] is not None]
        
        for cite_id in ids:
            for id,selected_paper in enumerate(selected_papers):
                cited_paper_id = cite_id["paper_id"]
                cited_paper_id = cited_paper_id.replace("paper_","")
                if cited_paper_id == "null":
                    continue
                if cited_paper_id == "None":
                    continue
                if "id" not in selected_paper:
                    continue
                try:
                    if int(cited_paper_id) == int(selected_paper["id"]):
                        cite_id["summary"] = selected_paper["abstract"]
                except Exception as e:
                    print(e)
                    continue
        quotes_with_cition_info = {quote: [] for quote in quotes}
        for quote in quotes:
            quote_has_match = False
            for cited_id in ids:
                c_text = cited_id["citation_text"]
                year = re.findall(r'(?<!\d)\d{4}(?!\d)',c_text)
                year = year[0] if year else None
                if "summary" not in cited_id:
                    continue
                if c_text in quote:
                    quote_has_match = True
                    quotes_with_cition_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["summary"])
                elif c_text.split(".")[0] in quote and year in quote:
                    quote_has_match = True
                    quotes_with_cition_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["summary"])
                elif c_text.split("(")[0] in quote and year in quote:
                    quote_has_match = True
                    quotes_with_cition_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["summary"])
                elif c_text.split(",")[0] in quote and year in quote:
                    quote_has_match = True
                    quotes_with_cition_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["summary"])
                elif c_text.split("et")[0] in quote and year in quote:
                    quote_has_match = True
                    quotes_with_cition_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["summary"])
                elif c_text.split("&")[0] in quote and year in quote:
                    quote_has_match = True
                    quotes_with_cition_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["summary"])
                elif c_text.split("and")[0] in quote and year in quote:
                    quote_has_match = True
                    quotes_with_cition_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["summary"])
        
            # 如果没有匹配，删除这个quote
            if not quote_has_match:
                del quotes_with_cition_info[quote]        
        try:
            yes_gemini = []
            no_gemini = []
            yes_deepseek = []
            no_deepseek = []
            for i,quote in enumerate(quotes_with_cition_info.keys()):
                if len(quotes_with_cition_info[quote]) == 0:
                    continue
                q = list(set(quotes_with_cition_info[quote]))
                source = "\n\n".join(q)
                # print(f"source: {source}")
                # print(f"quote: {quote}")
                score = judge_gemini._get_pair_score_new(source, quote)
                print(f"{i}. {score} by gemini")
                if score.lower() == "yes":
                    yes_gemini.append({"id": i, "claim": quote, "source": q})
                else:
                    no_gemini.append({"id": i, "claim": quote, "source": q})           
                
                score = judge_deepseek._get_pair_score_new(source, quote)
                print(f"{i}. {score} by deepseek")
                if score.lower() == "yes":
                    yes_deepseek.append({"id": i, "claim": quote, "source": q})
                else:
                    no_deepseek.append({"id": i, "claim": quote, "source": q})
                
                quotes_with_cition_info[quote] = q
            
            # 计算yes_ids和no_ids,两个模型都认为才算对,
            yes_gemini_ids = [(j["id"],tuple(j["source"])) for j in yes_gemini]
            yes_deepseek_ids = [(j["id"],tuple(j["source"])) for j in yes_deepseek]
            no_gemini_ids = [(j["id"],tuple(j["source"])) for j in no_gemini]
            no_deepseek_ids = [(j["id"],tuple(j["source"])) for j in no_deepseek]
            yes_ids = list(set(yes_gemini_ids) & set(yes_deepseek_ids))
            # 两个模型一个认为对，一个认为错，则算错
            no_ids = list(set(no_gemini_ids) | set(no_deepseek_ids))
            
            # 带引用的claim数量
            
            total_claims = len(quotes_with_cition_info.keys())
            # related work中总共的句子数量
            total_sentences = len(count_sentences(related_work))
            ## 1.claim_precision 正确引用的claim数量 / 所有claim数量（句子层面的）
            claim_precision = len(yes_ids) / total_claims
            claim_precision = round(claim_precision,3)
            
            ## 2.citation_precision 正确引用数量 / 所有引用数量 （引用层面的）
            correct_source = 0
            for yes_id in yes_ids:
                source = yes_id[1]
                source = list(source)
                print(f"source: {source}")
                print(f"len(source): {len(source)}")
                correct_source += len(source)
            print(f"correct_source: {correct_source}")
            
            
            total_citations = len(citations)
            print(f"total_citations: {total_citations}")
            citation_precision = correct_source / total_citations
            citation_precision = round(citation_precision,3)
            
            ## 3.reference_precision 被正确引用的不同论文篇数（其实就是正确的引用去重之后） / 参考文献总篇数 （信源层面的）
            correct_reference_source = set()
            for yes_id in yes_ids:
                source = yes_id[1]
                source = list(source)
                for s in source:
                    correct_reference_source.add(s)
            print(f"correct_reference_source: {correct_reference_source}")

            ## 全部被引用上的文章数量，将被引用的信源进行去重，然后计算数量
            unique_citations = set()
            for quote in quotes_with_cition_info.keys():
                for citation in quotes_with_cition_info[quote]:
                    unique_citations.add(citation)
            reference_precision = len(correct_reference_source) / len(selected_papers) # unique_citations
            reference_precision = round(reference_precision,3)
            
            ## 4. citation_density 引用总数 ÷ 正文句子总数(引用密度层面的)
            citation_density = total_citations / total_sentences
            citation_density = round(citation_density,3)
            
            ## 5. avg_citation_per_sentence 引用总数 ÷ claim总数(引用密度层面的)
            avg_citation_per_sentence = total_citations / total_claims
            avg_citation_per_sentence = round(avg_citation_per_sentence,3)
            
            
            item["yes_ids"] = yes_ids
            item["no_ids"] = no_ids
            item["yes_gemini_ids"] = yes_gemini_ids
            item["yes_deepseek_ids"] = yes_deepseek_ids
            item["no_gemini_ids"] = no_gemini_ids
            item["no_deepseek_ids"] = no_deepseek_ids
            item["yes_gemini"] = yes_gemini
            item["no_gemini"] = no_gemini
            item["yes_deepseek"] = yes_deepseek
            item["no_deepseek"] = no_deepseek
            item["claim_precision"] = claim_precision
            item["citation_precision"] = citation_precision
            item["reference_precision"] = reference_precision
            item["citation_density"] = citation_density
            item["avg_citation_per_sentence"] = avg_citation_per_sentence
            with jsonlines.open(output_file, "a") as writer:
                writer.write(item)
        except Exception as e:
            print(e)
            continue
        print(f"claim_precision: {claim_precision}")
        print(f"citation_precision: {citation_precision}")
        print(f"reference_precision: {reference_precision}")
        print(f"citation_density: {citation_density}")
        print(f"avg_citation_per_sentence: {avg_citation_per_sentence}")
        claim_precision_list.append(claim_precision)
        citation_precision_list.append(citation_precision)
        reference_precision_list.append(reference_precision)
        citation_density_list.append(citation_density)
        avg_citation_per_sentence_list.append(avg_citation_per_sentence)
        print(f"已完成第 {idx} 项")
    print(f"Claim Precision: {sum(claim_precision_list) / len(claim_precision_list)}")  
    print(f"Citation Precision: {sum(citation_precision_list) / len(citation_precision_list)}")
    print(f"Reference Precision: {sum(reference_precision_list) / len(reference_precision_list)}")
    print(f"Citation Density: {sum(citation_density_list) / len(citation_density_list)}")
    print(f"Avg Citation Per Sentence: {sum(avg_citation_per_sentence_list) / len(avg_citation_per_sentence_list)}")


input_file = "/home/liujian/project/2025-07/A2R-code-reproduction/results_baselines/naive_rag_gemini_related_work_cited_sentences.json"
output_file = "/home/liujian/project/2025-07/A2R-code-reproduction/results_baselines/naive_rag_gemini_related_work_cited_sentences_eval.jsonl"
eval_citation_baselines(input_file,output_file)







        
    
    
    
# process_with_checkpoint_extract_cited_sentences(input_file,output_file)
# jsonl2json(output_file,output_file.replace(".jsonl",".json"))

# with open("/home/liujian/project/2025-07/A2R-code-reproduction/results_baselines/naive_rag_gemini_related_work_cited_sentences.json","r",encoding="utf-8") as f:
#     data = json.load(f)


# claim_precision_list = []
# citation_precision_list = []
# reference_precision_list = []
# citation_density_list = []
# avg_citation_per_sentence_list = []

# for item in data[:]:
#     yes_count = []
#     related_work_info = item["related_work"]
#     selected_papers = item["reranked_results"]
#     related_work = item["related_work"]
#     sentences = len(count_sentences(related_work))
#     if "citations" not in item:
#         continue
#     cited_ids = item["citations"]
#     if len(cited_ids) == 0:
#         continue
#     print("quotes:")
#     quotes = item["quotes"]
#     if isinstance(quotes,str):
#         continue
#     print(quotes)
#     prompt = f"""
#     Task:
#     You are a helpful assistant that can extract the citations from the text.
#     You will be given a list of quotes and a list of selected papers and their citations.
#     First, match each provided quote to its correct source (research paper) based on:
#     Explicit mentions of author names and years (e.g., "Han et al. (2020)")

#     Unique method/framework names (e.g., "TextAttack", "DILMA")

#     Direct conceptual alignment (e.g., a quote about "adversarial training for POS tagging" → Yasunaga et al. 2017)

#     For each matched pair, strictly assess whether the quote is fully supported by the source.
    
#     Strict Evaluation Rules (Automatic "No" if any violation,the ratio of yes to no is approximately controlled at 6:4)
#     A quote is unsupported if:

#     Author/Year Mismatch: The quote cites incorrect authors or years vs. the source.

#     Title/Concept Misalignment: The quote’s description contradicts or ambiguously paraphrases the source’s title/core contribution.

#     Example: Quote says "analyzes n-gram frequency patterns" → Source title says "n-gram Frequency Descend" → No (phrasing mismatch).

#     Unverified Method Names: The quote names a method (e.g., "ATINTER") not explicitly mentioned in the source title.

#     Example: Source title is Don’t Retrain, Just Rewrite → Quote mentions "ATINTER" → No (name absent in title).
    
#     Below is the list of quotes and the list of selected papers and their citations.
#     """
#     quotes_str = ""
#     for id,quote in enumerate(quotes):
#         quotes_str += f"Quote {id+1}: {quote}\n"
#     print(quotes_str)
#     selected_papers_str = ""
#     for id,selected_paper in enumerate(selected_papers):
#         summary = selected_paper["abstract"]
#         citation = selected_paper["reference"]
#         selected_papers_str += f"Selected Paper {id+1}:\nSummary: {summary}\nCitation: {citation}\n"

#     prompt += quotes_str + "\n" + selected_papers_str
    
        
#     prompt += """
#     Please only output a json object with the following format:
#     {
#         "correct_quote_ids": [1,2,3]
#     }
#     """
#     response = client.get_completion(prompt)
#     print(response)
#     response = json_repair.loads(response)
#     yes_count = len(response["correct_quote_ids"])
    
#     # 1. claim precision
#     claim_precision = yes_count / len(quotes)
#     claim_precision_list.append(round(claim_precision,3))
#     print(f"Claim Precision: {claim_precision}")
#     # 2. citation precision
#     citation_precision = yes_count / len(cited_ids)
#     citation_precision_list.append(round(citation_precision,3))
#     print(f"Citation Precision: {citation_precision}")
    
#     # 3. reference precision
#     reference_precision = yes_count / len(selected_papers)
#     reference_precision_list.append(round(reference_precision,3))
#     print(f"Reference Precision: {reference_precision}")
#     # 4. citation_density 引用总数 ÷ 正文句子总数(引用密度层面的)
#     citation_density = len(cited_ids) / len(count_sentences(related_work))
#     citation_density_list.append(round(citation_density,3))
#     print(f"Citation Density: {citation_density}")
#     # 5. avg_citation_per_sentence = total_citations / total_claims
#     avg_citation_per_sentence = len(cited_ids) / len(quotes)
#     avg_citation_per_sentence_list.append(round(avg_citation_per_sentence,3))
#     print(f"Avg Citation Per Sentence: {avg_citation_per_sentence}")

# print(f"Claim Precision: {sum(claim_precision_list) / len(claim_precision_list)}")
# print(f"Citation Precision: {sum(citation_precision_list) / len(citation_precision_list)}")
# print(f"Reference Precision: {sum(reference_precision_list) / len(reference_precision_list)}")
# print(f"Citation Density: {sum(citation_density_list) / len(citation_density_list)}")
# print(f"Avg Citation Per Sentence: {sum(avg_citation_per_sentence_list) / len(avg_citation_per_sentence_list)}")
