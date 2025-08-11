import json_repair
from evaluation.agents.judge import Judge
from citegeist.utils.infer import jsonl2json
import json,jsonlines

from citegeist.utils.infer import load_processed_ids
from tqdm import tqdm
import os
print(os.getenv("OPENAI_API_KEY"))
import dotenv
dotenv.load_dotenv()
judge = Judge(
    model = "openai/o3-mini"
)

def process_simple_eval(abstract,related_work):
    result = judge.evaluate_language(related_work, abstract)
    print(result)
    print(f"Start batch_criteria_based_judging")
    criterion = ["Structure", "Relevance","Coverage"]
    scores = judge.batch_criteria_based_judging(related_work, abstract, criterion)
    print(scores)
    
def process_with_checkpoint_eval_for_text_quality_baselines(input_file):
    with open(input_file,"r",encoding="utf-8") as f:
        data = json.load(f)

    clarity = []
    structure = []
    relevance = []
    motivation = []
    critics = []
    for item in tqdm(data, desc="Processing items"):
        
        related_work = item["related_work"]
        print(related_work)
        abstract = item["abstract"]
        related_work += "\n" + "## Citations Quality for the related work" + "\n"
        result = judge.evaluate_text_quality(related_work, abstract)
        result = result.replace("```json","").replace("```","")
        result = json_repair.loads(result)
        result = {
            "Clarity": result["Clarity"]["score"],
            "Structure": result["Structure"]["score"],
            "Relevance": result["Relevance"]["score"],
            "Motivation": result["Motivation"]["score"],
            "Critics": result["Critics"]["score"]
        }
        clarity.append(result["Clarity"])
        structure.append(result["Structure"])
        relevance.append(result["Relevance"])
        motivation.append(result["Motivation"])
        critics.append(result["Critics"])
        
        print(f"Clarity: {result['Clarity']}")
        print(f"Structure: {result['Structure']}")
        print(f"Relevance: {result['Relevance']}")
        print(f"Motivation: {result['Motivation']}")
        print(f"Critics: {result['Critics']}")
        print("-"*100)
        item["text_quality"] = result
    average_clarity = sum(clarity) / len(clarity)
    average_structure = sum(structure) / len(structure)
    average_relevance = sum(relevance) / len(relevance)
    average_motivation = sum(motivation) / len(motivation)
    average_critics = sum(critics) / len(critics)
    print(f"Average clarity: {average_clarity}")
    print(f"Average structure: {average_structure}")
    print(f"Average relevance: {average_relevance}")
    print(f"Average motivation: {average_motivation}")
    print(f"Average critics: {average_critics}")
    print("MACG done")
    with open(input_file.replace(".json","_text_quality.json"),"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=4)
    
def process_with_checkpoint_eval_for_text_quality(input_file):
    with open(input_file,"r",encoding="utf-8") as f:
        data = json.load(f)

    clarity = []
    structure = []
    relevance = []
    motivation = []
    critics = []
    for item in tqdm(data, desc="Processing items"):
        
        related_work = item["related_work"]["related_works"]
        print(related_work)
        abstract = item["abstract"]
        related_work += "\n" + "## Citations Quality for the related work" + "\n"
        result = judge.evaluate_text_quality(related_work, abstract)
        result = result.replace("```json","").replace("```","")
        result = json_repair.loads(result)
        result = {
            "Clarity": result["Clarity"]["score"],
            "Structure": result["Structure"]["score"],
            "Relevance": result["Relevance"]["score"],
            "Motivation": result["Motivation"]["score"],
            "Critics": result["Critics"]["score"]
        }
        clarity.append(result["Clarity"])
        structure.append(result["Structure"])
        relevance.append(result["Relevance"])
        motivation.append(result["Motivation"])
        critics.append(result["Critics"])
        
        print(f"Clarity: {result['Clarity']}")
        print(f"Structure: {result['Structure']}")
        print(f"Relevance: {result['Relevance']}")
        print(f"Motivation: {result['Motivation']}")
        print(f"Critics: {result['Critics']}")
        print("-"*100)
        item["text_quality"] = result
    average_clarity = sum(clarity) / len(clarity)
    average_structure = sum(structure) / len(structure)
    average_relevance = sum(relevance) / len(relevance)
    average_motivation = sum(motivation) / len(motivation)
    average_critics = sum(critics) / len(critics)
    print(f"Average clarity: {average_clarity}")
    print(f"Average structure: {average_structure}")
    print(f"Average relevance: {average_relevance}")
    print(f"Average motivation: {average_motivation}")
    print(f"Average critics: {average_critics}")
    with open(input_file.replace(".json","_text_quality.json"),"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=4)
    
def process_with_checkpoint_eval_for_vallina_gpt(input_file, output_file):
    with open(input_file,"r",encoding="utf-8") as f:
        data = json.load(f)
        
    processed_ids = load_processed_ids(output_file)
    print(f"已处理 {len(processed_ids)} 个项目，从第 {len(processed_ids)} 个开始继续...")
    
    # 过滤出未处理的数据
    remaining_data = []
    for item in data[:]:
        if item["id"] not in processed_ids:
            remaining_data.append(item)
            
    print(f"剩余 {len(remaining_data)} 个项目")
    language = []
    critical = []
    structure = []
    relevance = []
    coverage = []
    for item in tqdm(remaining_data, desc="Processing items"):
        try:
            
            abstract = item["abstract"]
            related_work = item["related_work"]
            
            print(f"Start evaluating critical dimensions")
            result = judge.evaluate_critical(related_work, abstract)
            critical.append(result)
            print(result)
            print(f"Start evaluating language dimension")
            result = judge.evaluate_language(related_work, abstract)
            language.append(result)
            print(result)
            print(f"Start batch_criteria_based_judging")
            criterion = ["Structure", "Relevance","Coverage"]
            scores = judge.batch_criteria_based_judging(related_work, abstract, criterion)
            structure.append(scores[0])
            relevance.append(scores[1])
            coverage.append(scores[2])
            print(scores)
            
        
        except Exception as e:
            print(e)
            continue
    average_language = sum(language) / len(language)
    average_critical = sum(critical) / len(critical)
    average_structure = sum(structure) / len(structure)
    average_relevance = sum(relevance) / len(relevance)
    average_coverage = sum(coverage) / len(coverage)
    print(f"Average language: {average_language}")
    print(f"Average critical: {average_critical}")
    print(f"Average structure: {average_structure}")
    print(f"Average relevance: {average_relevance}")
    print(f"Average coverage: {average_coverage}")
    
    return average_language,average_critical,average_structure,average_relevance,average_coverage
        

def process_with_checkpoint_eval(input_file, output_file):
    with open(input_file,"r",encoding="utf-8") as f:
        data = json.load(f)
        
    processed_ids = load_processed_ids(output_file)
    print(f"已处理 {len(processed_ids)} 个项目，从第 {len(processed_ids)} 个开始继续...")
    
    # 过滤出未处理的数据
    remaining_data = []
    for item in data[:]:
        if item["id"] not in processed_ids:
            remaining_data.append(item)
            
    print(f"剩余 {len(remaining_data)} 个项目")
    
    for item in tqdm(remaining_data, desc="Processing items"):
        try:
            language = []
            critical = []
            structure = []
            relevance = []
            coverage = []
            abstract = item["abstract"]
            related_work = item["related_work"]["related_works"]
            
            print(f"Start evaluating critical dimensions")
            result = judge.evaluate_critical(related_work, abstract)
            critical.append(result)
            print(result)
            print(f"Start evaluating language dimension")
            result = judge.evaluate_language(related_work, abstract)
            language.append(result)
            print(result)
            print(f"Start batch_criteria_based_judging")
            criterion = ["Structure", "Relevance","Coverage"]
            scores = judge.batch_criteria_based_judging(related_work, abstract, criterion)
            structure.append(scores[0])
            relevance.append(scores[1])
            coverage.append(scores[2])
            print(scores)
            
        
        except Exception as e:
            print(e)
            continue
    print(language)
    print(critical)
    print(structure)
    print(relevance)
    print(coverage)
    
    average_language = sum(language) / len(language)
    average_critical = sum(critical) / len(critical)
    average_structure = sum(structure) / len(structure)
    average_relevance = sum(relevance) / len(relevance)
    average_coverage = sum(coverage) / len(coverage)
    print(f"Average language: {average_language}")
    print(f"Average critical: {average_critical}")
    print(f"Average structure: {average_structure}")
    print(f"Average relevance: {average_relevance}")
    print(f"Average coverage: {average_coverage}")
    
    return average_language,average_critical,average_structure,average_relevance,average_coverage
        
def main_MACG():
    process_with_checkpoint_eval_for_text_quality(
        "/home/liujian/project/2025-07/A2R-code-reproduction/results_our_workflow/citation_eval/result_our_workflow_eval_classify_errors_refine_extract_cited_sentences.json"
    )
    print(f"ourworkflow finished")


def main_perplexity():
    process_with_checkpoint_eval_for_text_quality(
        "/home/liujian/project/2025-07/A2R-code-reproduction/results_baselines/perplexity_deep_research_related_work_cited_sentences_with_cition_info.json"
    )
    print(f"perplexity finished")

def main_naive_rag_gemini():
    process_with_checkpoint_eval_for_text_quality_baselines(
        "/home/liujian/project/2025-07/A2R-code-reproduction/results_baselines/naive_rag_gemini_related_work_cited_sentences.json"
    )
    print(f"naive_rag_gemini finished")

def main_gpt_4o_mini_search():
    process_with_checkpoint_eval_for_text_quality_baselines(
        "/home/liujian/project/2025-07/A2R-code-reproduction/results_baselines/vallina_gpt_related_work_cited_sentences.json"
    )
    print(f"gpt_4o_mini_search finished")
def main_citegeist():
    process_with_checkpoint_eval_for_text_quality_baselines(
        "/home/liujian/project/2025-07/A2R-code-reproduction/results_baselines/citegeist/result_citegeist_extract_cited_sentences.json"
    )
    print(f"citegeist finished")
def main_without_fact_check():
    process_with_checkpoint_eval_for_text_quality(
        "/home/liujian/project/2025-07/A2R-code-reproduction/ablation_experiment/result_without_fact_check.json"
    )
    print(f"without_fact_check finished")

def main_without_feedback_revision():
    process_with_checkpoint_eval_for_text_quality(
        "/home/liujian/project/2025-07/A2R-code-reproduction/ablation_experiment/result_without_feedback_revision_1.json"
    )
    print(f"without_feedback_revision finished")
    
def main_with_DAG():
    process_with_checkpoint_eval_for_text_quality(
        "/home/liujian/project/2025-07/A2R-code-reproduction/ablation_experiment/result_without_DAG_1.json"
    )
    print("without_DAG finished")
    
def main_without_summarization():
    process_with_checkpoint_eval_for_text_quality(
        "/home/liujian/project/2025-07/A2R-code-reproduction/ablation_experiment/result_without_summarization_with_cite_ids.json"
    )
    print(f"without_summarization finished")
    
if __name__ == "__main__":
    # main_citegeist()
    # main_naive_rag_gemini()
    # main_without_fact_check()
    # main_without_feedback_revision()
    # main_MACG()
    # main_perplexity()
    # main_gpt_4o_mini_search()
    # main_with_DAG()
    main_without_summarization()
# perplexity_deep_research_eval

# perplexity_deep_research_eval = "/home/liujian/project/2025-07/A2R-code-reproduction/results_baselines/perplexity_deep_research_related_work.jsonl"
# jsonl2json(perplexity_deep_research_eval,perplexity_deep_research_eval.replace(".jsonl",".json"))
# perplexity_deep_research_eval = "/home/liujian/project/2025-07/A2R-code-reproduction/results_baselines/perplexity_deep_research_related_work.json"
# jsonl2json("/home/liujian/project/2025-07/A2R-code-reproduction/citegeist/result_citegeist_3_sections.jsonl","/home/liujian/project/2025-07/A2R-code-reproduction/citegeist/result_citegeist_3_sections.json")
# jsonl2json("/home/liujian/project/2025-07/A2R-code-reproduction/citegeist/result_citegeist.jsonl","/home/liujian/project/2025-07/A2R-code-reproduction/citegeist/result_citegeist_3.json")
# with open("/home/liujian/project/2025-07/A2R-code-reproduction/citegeist/result_citegeist_3.json","r",encoding="utf-8") as f1,open("/home/liujian/project/2025-07/A2R-code-reproduction/citegeist/result_citegeist_3_sections.json","r",encoding="utf-8") as f2:
#     data_1 = json.load(f1)
#     data_2 = json.load(f2)

# data = data_1 + data_2

# for i,item in enumerate(data):
#     item["id"] = i
# with open("/home/liujian/project/2025-07/A2R-code-reproduction/citegeist/result_citegeist.json","w",encoding="utf-8") as f:
#     json.dump(data,f,ensure_ascii=False,indent=4)

# process_with_checkpoint_eval_for_text_quality(
#     "/home/liujian/project/2025-07/A2R-code-reproduction/results_our_workflow/citation_eval/result_our_workflow_eval_classify_errors_refine_extract_cited_sentences.json"
# )
# print(f"ourworkflow finished")
# /home/liujian/project/2025-07/A2R-code-reproduction/ourworkflow/result_all_with_cite_07_18_test_post_hoc_refinement_eval_1.json
# /home/liujian/project/2025-07/A2R-code-reproduction/results_baselines/perplexity_deep_research_related_work.json
#/home/liujian/project/2025-07/A2R-code-reproduction/ourworkflow/result_all_with_cite_2_quotes_split_by_gemini_eval_refine_classify_errors_refine_extract_related_work_citations.json
# process_with_checkpoint_eval(
#     "/home/liujian/project/2025-07/A2R-code-reproduction/ourworkflow/result_all_with_cite_07_18_test_post_hoc_refinement_eval_1.json",
#     "/home/liujian/project/2025-07/A2R-code-reproduction/ourworkflow/result_all_with_cite_2_quotes_split_by_gemini_eval_refine_classify_errors_refine_extract_related_work_citations_critical_eval.json"
# )


# abstract = """
# "<|reference_start|>Flow of Reasoning:Training LLMs for Divergent Problem Solving with Minimal Examples: The ability to generate diverse solutions to a given problem is a hallmark of human creativity. This divergent reasoning is also crucial for machines, enhancing their robustness and enabling them to assist humans in many applications such as scientific discovery. However, existing approaches to multi-step reasoning with large language models (LLMs) have mostly focused only on reasoning accuracy, without further discovering more diverse valid solutions. For example, supervised fine-tuning can improve LLM reasoning quality, but requires extensive supervised data to capture the full range of possible solutions. Reinforcement learning aims to find limited highest-reward solutions while neglecting the solution diversity. To fill this gap, we propose Flow of Reasoning (FoR), an efficient diversity-seeking LLM finetuning method aimed at improving reasoning quality and diversity with minimal data. FoR formulates multi-step LLM reasoning as a Markovian flow on a DAG-structured reasoning graph. This formulation allows us to incorporate and adapt principled GFlowNet approaches, for finetuning LLMs to sample diverse reasoning paths with probabilities proportional to the (unnormalized) reward of target problems. Extensive experiments show that, with limited training examples (e.g., 15 examples), FoR enables the discovery of diverse, creative, high-quality solutions, greatly outperforming a wide range of existing inference and training methods across five challenging puzzle-solving tasks, including BlocksWorld (embodied reasoning), Game24 (math puzzle solving), Rubik's Cube (spatial reasoning), 1D-ARC (abstraction reasoning), and PrOntoQA (logical reasoning). Code is available at https://github.com/Yu-Fangxu/FoR.<|reference_end|>"
# """
# related_work = """
# ### LLM Reasoning Paradigms and Tasks

# Large Language Models (LLMs) struggle with complex reasoning tasks, particularly those requiring multi-step thinking and diverse skill integration <cite>(Yao et al., 2024; Chia et al., 2024)</cite>. A key limitation is their lack of an internal world model for state prediction and long-term outcome simulation, hindering deliberate planning <cite>(Hao et al., 2023)</cite>. Existing approaches primarily focus on reasoning accuracy, often neglecting solution diversity <cite>(Yu et al., 2024)</cite>. Supervised fine-tuning improves reasoning quality but requires extensive labeled data, while reinforcement learning over-exploits high-reward solutions at the expense of diversity <cite>(Yu et al., 2024; Ho et al., 2024)</cite>. Techniques like STaR (Self-Taught Reasoner) iteratively refine reasoning by generating and correcting rationales, but focus primarily on accuracy rather than diversity <cite>(Zelikman et al., 2022)</cite>.

# Methods like Reasoning Paths Optimization (RPO) and HDFlow optimize reasoning paths for accuracy but target single optimal solutions <cite>(Chia et al., 2024; Yao et al., 2024)</cite>. DOTS dynamically selects reasoning trajectories but remains constrained by predefined actions <cite>(Yue et al., 2024)</cite>. These approaches fail to address the need for diverse reasoning paths, limiting their applicability to open-ended problems. Marco-o1 and CPL explore open-ended resolutions and generalization but rely on computationally expensive search mechanisms <cite>(Zhao et al., 2024; Wang et al., 2024)</cite>. ReGenesis synthesizes diverse reasoning paths but struggles with minimal-data scenarios <cite>(Peng et al., 2024)</cite>. Benchmarks like ProcBench and CLR-Fact highlight LLMs' reasoning deficiencies, underscoring the need for methods that balance accuracy and diversity <cite>(Fujisawa et al., 2024; Zheng et al., 2024)</cite>.

# Recent work has also highlighted challenges in reasoning faithfulness, where LLMs generate intermediate steps that do not reliably lead to their final answers <cite>(Paul et al., 2024; Creswell & Shanahan, 2022)</cite>. Frameworks like FRODO and Faithful CoT address this by ensuring reasoning chains are valid and interpretable <cite>(Paul et al., 2024; Lyu et al., 2023)</cite>, while others investigate how easily irrelevant inputs can skew LLM responses <cite>(Wu et al., 2024)</cite>. The interpretability of reasoning explanations has also been scrutinized, with studies showing that current prompting techniques vary widely in their robustness and utility <cite>(Yeo et al., 2024)</cite>.

# ### Methods for Diverse and Efficient LLM Reasoning

# The limitations of accuracy-focused methods create a gap for approaches that prioritize diversity. Flow of Reasoning (FoR) addresses this by formulating reasoning as a Markovian flow on a DAG-structured graph, enabling diverse path sampling with minimal data <cite>(Yu et al., 2024)</cite>. Unlike reward-maximizing RL, which over-exploits high-reward actions <cite>(Ho et al., 2024)</cite>, FoR leverages GFlowNets to sample proportionally to reward, building on foundational work in non-iterative diverse candidate generation <cite>(Bengio et al., 2021)</cite>. This approach contrasts with tree-based methods like RAP and ToT, which focus on strategic exploration for optimal solutions but face efficiency challenges <cite>(Hao et al., 2023; Yao et al., 2024; Chen et al., 2024)</cite>.

# SWAP and "Unleashing the Creative Mind" encourage diversity but focus on accuracy or hierarchical tactics <cite>(Xiong et al., 2024; Ling et al., 2023)</cite>. DCoT generates diverse reasoning chains but requires extensive fine-tuning <cite>(Puerto et al., 2024)</cite>. SIKeD mitigates bias in smaller models but doesn't address diversity in reasoning paths <cite>(Adarsh et al., 2024)</cite>. DIV-SE and IDIV-SE vary input prompts rather than reasoning paths <cite>(Naik et al., 2023)</cite>. Automatic CoT methods like Auto-CoT improve reasoning by diversifying demonstrations, but remain constrained to prompting rather than fine-tuning <cite>(Zhang et al., 2022)</cite>.

# FoR uniquely combines efficiency and diversity by adapting GFlowNets for LLM fine-tuning, outperforming existing methods in generating high-quality, creative solutions with minimal data <cite>(Yu et al., 2024)</cite>. Its DAG-structured formulation avoids the search space limitations of methods like BoN <cite>(Wu et al., 2024)</cite>, making it particularly effective for complex reasoning tasks. Recent work on unfaithful reasoning in LLMs further underscores the importance of approaches like FoR that ensure diverse solutions are both valid and interpretable <cite>(Li et al., 2024)</cite>.
# """
# process_simple_eval(abstract,related_work)