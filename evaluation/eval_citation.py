import os
import json
import json_repair
from agents.judge import Judge
from citegeist.utils.llm_clients.gemini_client import GeminiClient
from citegeist.utils.prompts import (
    process_data_for_extract_cited_sentences,
    process_data_for_classify_errors,
    process_data_for_count_citations
)

llm_client = GeminiClient(
    api_key = os.environ.get("OPENROUTER_API_KEY", ""),
    model_name="google/gemini-2.5-flash-preview-09-2025")
judge_openai = Judge(model="openai/gpt-5-nano")

## 需要related work和cite_ids
## related work中切出带引用的claims
## cite_ids 映射到selected_papers

## 构造quotes_with_citation_info
import re

def count_sentences(text):
    sentences = re.split(r"[.!?\n]+(?:\s|\n|$)", text.strip())
    sentences = [s for s in sentences if s]
    return sentences

def build_citation_verification_data(
        quotes: list[str], 
        selected_papers: list[dict], 
        cite_ids: list[dict],
        use_abstract: bool = False
    ):
        """构建引用验证所需的数据结构"""
        quotes_with_citation_info = {quote: [] for quote in quotes}
        ids = [i for i in cite_ids if "paper_id" in i and i["paper_id"] is not None]

        for cited_id in ids:
            for _,selected_paper in enumerate(selected_papers):
                cited_paper_id = cited_id["paper_id"]
                cited_paper_id = cited_paper_id.replace("paper_","")
                if cited_paper_id == "null":
                    continue
                if "cite_ids" not in selected_paper:
                    continue
                # if "paper_id" not in selected_paper:
                #     continue
                try:
                    if int(cited_paper_id) == int(selected_paper["cite_ids"][0])+1:
                        
                        if use_abstract:
                            cited_id["abstract"] = selected_paper["abstract"]
                        else:
                            cited_id["summary"] = selected_paper["summary"]
                except Exception as e:
                    print(e)
                    continue
        for quote in quotes:
            for cited_id in ids:
                c_text = cited_id["citation_text"]
                year = re.findall(r'(?<!\d)\d{4}(?!\d)',c_text)
                year = year[0] if year else None
                if "summary" not in cited_id:
                    continue
                if use_abstract:
                    evidence = cited_id["abstract"]
                else:
                    evidence = cited_id["summary"]
                if c_text in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + evidence)
                elif c_text.split(".")[0] in quote and year in quote:   
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + evidence)
                elif c_text.split("(")[0] in quote and year in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + evidence)
                elif c_text.split(",")[0] in quote and year in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + evidence)
                elif c_text.split("et")[0] in quote and year in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + evidence)
                elif c_text.split("&")[0] in quote and year in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + evidence)
                elif c_text.split("and")[0] in quote and year in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + evidence)
        print("quotes_with_citation_info:")
        print(quotes_with_citation_info)
        return quotes_with_citation_info

def eval_citation(related_work,cite_ids, selected_papers):
    prompt = process_data_for_extract_cited_sentences(related_work)
    cited_sentences = llm_client.get_completion(prompt)
    cited_sentences = json_repair.loads(cited_sentences)
    quotes_with_citation_info = build_citation_verification_data(
            cited_sentences, selected_papers, cite_ids, use_abstract=True
        )
    
    true_ids = []
    false_ids = []

    for i,quote in enumerate(quotes_with_citation_info.keys()):
        if len(quotes_with_citation_info[quote]) == 0:
            continue
        q = list(set(quotes_with_citation_info[quote]))
        source = "\n".join(q)
        print(f"source: {source}")
        print(f"quote: {quote}")
        score = judge_openai.get_pair_score_new(source, quote)
        print(f"{i}. {score}")
        if score.lower() == "yes":
            true_ids.append({"id": i, "claim": quote, "source": source})
        else:
            false_ids.append({"id": i, "claim": quote, "source": source})
    total_claims = len(quotes_with_citation_info.keys())
    claim_precision = len(true_ids) / total_claims
    claim_precision = round(claim_precision,3)
    
    
    total_citations = len(cite_ids)
    citation_precision = len(true_ids) / total_citations
    citation_precision = round(citation_precision,3)
    
    
    total_sentences = len(count_sentences(related_work))
    citation_density = total_citations / total_sentences
    citation_density = round(citation_density,3)
    
    avg_citation_per_sentence = total_citations / total_claims
    avg_citation_per_sentence = round(avg_citation_per_sentence,3)
    
    citation_quality = {
        "claim_precision": claim_precision,
        "citation_precision": citation_precision,
        "citation_density": citation_density,
        "avg_citation_per_sentence": avg_citation_per_sentence
    }
    print(f"citation_quality: {citation_quality}")
    
    
    direct_contradiction = 0
    information_not_present = 0
    misrepresentation = 0
    incorrect_attribution = 0
    other = 0
    
    for id in false_ids:
        source = id["source"]
        claim = id["claim"]
        prompt = process_data_for_classify_errors(claim,source)
        result = llm_client.get_completion(prompt)
        result = json_repair.loads(result)
        if "Direct Contradiction" in result["error_type"]:
            direct_contradiction += 1
        elif "Information Not Present / Unsubstantiated" in result["error_type"]:
            information_not_present += 1
        elif "Misrepresentation / Imprecise Wording" in result["error_type"]:
            misrepresentation += 1
        elif "Incorrect Attribution" in result["error_type"]:
            incorrect_attribution += 1
        else:
            other += 1
        errors_count = {
            "direct_contradiction": direct_contradiction,
            "information_not_present": information_not_present,
            "misrepresentation": misrepresentation,
            "incorrect_attribution": incorrect_attribution,
            "other": other
        }
    print(f"errors_count: {errors_count}")
        
    
    return {
        "claim_precision": claim_precision,
        "citation_precision": citation_precision,
        "citation_density": citation_density,
        "avg_citation_per_sentence": avg_citation_per_sentence,
        "direct_contradiction": direct_contradiction,
        "information_not_present": information_not_present,
        "misrepresentation": misrepresentation,
        "incorrect_attribution": incorrect_attribution,
        "other": other
    }
    

# 对litllm生成的related work切分句子
def split_sentences(text):
    # 使用字典存储每个句子的引用信息
    pattern = r'[^.]*@cite_\d+[^.]*\.'
    matches = re.findall(pattern, text)

    results = []
    for sent in matches:
        # 提取该句子中所有的引用编号
        citations = re.findall(r'@cite_(\d+)', sent)
        results.append({
            'sentence': sent.strip(),
            'citations': citations,
            'citation_pattern': ', '.join([f'@cite_{c}' for c in citations])
        })
        
    return results

def eval_citation_fast_macg(related_work,cite_ids,quotes_with_citation_info):
    true_ids = []
    false_ids = []
    quote_no_source = []
    for i,quote in enumerate(quotes_with_citation_info.keys()):
        if len(quotes_with_citation_info[quote]) == 0:
            quote_no_source.append({"id": i, "claim": quote, "source": source})
            continue
        q = list(set(quotes_with_citation_info[quote]))
        source = "\n".join(q)
        score = judge_openai.get_pair_score_new(source, quote)
        print(f"{i}. {score}")
        if score.lower() == "yes":
            true_ids.append({"id": i, "claim": quote, "source": source})
        elif score.lower() == "no":
            false_ids.append({"id": i, "claim": quote, "source": source})
    total_claims = len(quotes_with_citation_info.keys()) - len(quote_no_source)
    print(f"total_claims: {total_claims}")
    print(f"len(true_ids): {len(true_ids)}")
    print(f"len(false_ids): {len(false_ids)}")
    claim_precision = len(true_ids) / total_claims
    claim_precision = round(claim_precision,3)
    
    prompt_count_citations = process_data_for_count_citations(related_work)
    count_citations = llm_client.get_completion(prompt_count_citations)
    count_citations = json_repair.loads(count_citations)
    total_citations = count_citations["number_of_citations"]
    citation_precision = len(true_ids) / total_citations
    citation_precision = round(citation_precision,3)
    
    
    total_sentences = len(count_sentences(related_work))
    citation_density = total_citations / total_sentences
    citation_density = round(citation_density,3)
    
    avg_citation_per_sentence = total_citations / total_claims
    avg_citation_per_sentence = round(avg_citation_per_sentence,3)
    
    citation_quality = {
        "claim_precision": claim_precision,
        "citation_precision": citation_precision,
        "citation_density": citation_density,
        "avg_citation_per_sentence": avg_citation_per_sentence
    }
    
    
    direct_contradiction = 0
    information_not_present = 0
    misrepresentation = 0
    incorrect_attribution = 0
    other = 0
    
    if len(false_ids) > 0:
        for id in false_ids:
            source = id["source"]
            claim = id["claim"]
            prompt = process_data_for_classify_errors(claim,source)
            result = llm_client.get_completion(prompt)
            result = json_repair.loads(result)
            if "Direct Contradiction" in result["error_type"]:
                direct_contradiction += 1
            elif "Information Not Present / Unsubstantiated" in result["error_type"]:
                information_not_present += 1
            elif "Misrepresentation / Imprecise Wording" in result["error_type"]:
                misrepresentation += 1
            elif "Incorrect Attribution" in result["error_type"]:
                incorrect_attribution += 1
            else:
                other += 1
            errors_count = {
                "direct_contradiction": direct_contradiction,
                "information_not_present": information_not_present,
                "misrepresentation": misrepresentation,
                "incorrect_attribution": incorrect_attribution,
                "other": other
            }
        print(f"errors_count: {errors_count}")
        
    
    return {
        "claim_precision": claim_precision,
        "citation_precision": citation_precision,
        "citation_density": citation_density,
        "avg_citation_per_sentence": avg_citation_per_sentence,
        "direct_contradiction": direct_contradiction,
        "information_not_present": information_not_present,
        "misrepresentation": misrepresentation,
        "incorrect_attribution": incorrect_attribution,
        "other": other
    }
    

# 对litllm生成的related work切分句子
def split_sentences(text):
    # 使用字典存储每个句子的引用信息
    pattern = r'[^.]*@cite_\d+[^.]*\.'
    matches = re.findall(pattern, text)

    results = []
    for sent in matches:
        # 提取该句子中所有的引用编号
        citations = re.findall(r'@cite_(\d+)', sent)
        results.append({
            'sentence': sent.strip(),
            'citations': citations,
            'citation_pattern': ', '.join([f'@cite_{c}' for c in citations])
        })
        
    return results
    
    
# data = {
#     "abstract": "",
#     "related_work": "",
#     "ref_abstract": {
#         "cite_N": [
#             0,1
#         ],
#         "abstracts": [
#             ""
#         ]
#     }
# }

def eval_autosurvey(data,related_work):
    true_ids = []
    false_ids = []
    for i,item in enumerate(data):
        sentence = item["sentence"]
        source = item["source"]
        score = judge_openai.get_pair_score_new(source, sentence)
        print(f"source: {source}")
        print(f"sentence: {sentence}")
        print(f"{i}. {score}")
        if score.lower() == "yes":
            true_ids.append({"id": i, "claim": sentence, "source": source})
        else:
            false_ids.append({"id": i, "claim": sentence, "source": source})
    total_claims = len(data)
    claim_precision = len(true_ids) / total_claims
    claim_precision = round(claim_precision,3)
    
    total_sentences = len(count_sentences(related_work))
    citation_density = total_claims / total_sentences
    citation_density = round(citation_density,3)
    
    avg_citation_per_sentence = total_claims / total_claims
    avg_citation_per_sentence = round(avg_citation_per_sentence,3)
    
    direct_contradiction = 0
    information_not_present = 0
    misrepresentation = 0
    incorrect_attribution = 0
    other = 0
    
    if len(false_ids) > 0:
        for id in false_ids:
            source = id["source"]
            claim = id["claim"]
            prompt = process_data_for_classify_errors(claim, source)
            result_str = llm_client.get_completion(prompt)
            
            try:
                result = json_repair.loads(result_str)
                
                # 验证数据结构
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]  # 取第一个元素
                
                if not isinstance(result, dict):
                    print(f"⚠️ 无效结果类型: {type(result)}")
                    print(f"内容: {result_str}")
                    continue
                    
                if "error_type" not in result:
                    print(f"⚠️ 缺少 error_type 字段")
                    print(f"完整结果: {result}")
                    continue
                    
                error_type = result["error_type"]
                
                # 错误分类逻辑
                if "Direct Contradiction" in error_type:
                    direct_contradiction += 1
                elif "Information Not Present / Unsubstantiated" in error_type:
                    information_not_present += 1
                elif "Misrepresentation / Imprecise Wording" in error_type:
                    misrepresentation += 1
                elif "Incorrect Attribution" in error_type:
                    incorrect_attribution += 1
                else:
                    other += 1
                    
            except Exception as e:
                print(f"❌ 处理错误时发生异常: {str(e)}")
                print(f"原始响应: {result_str}")
                
    result = {
        "claim_precision": claim_precision,
        "citation_density": citation_density,
        "avg_citation_per_sentence": avg_citation_per_sentence,
        "direct_contradiction": direct_contradiction,
        "information_not_present": information_not_present,
        "misrepresentation": misrepresentation,
        "incorrect_attribution": incorrect_attribution,
        "other": other
    }
    
    return result
    
def eval_litllm(data,related_work):
    results = split_sentences(related_work)
    true_ids = []
    false_ids = []
    ref_abstract = data["ref_abstract"]
    abstracts = ref_abstract["abstract"]
    for id in range(len(results)):
        claim = results[id]["sentence"]
        ids = results[id]["citations"]
        source = ""
        for i in ids:
            cited = "cited_text:" + abstracts[int(i)]
            source += cited
        score = judge_openai.get_pair_score_new(source, claim)
        print(f"source: {source}")
        print(f"claim: {claim}")
        print(f"{id}. {score}")
        
        if score.lower() == "yes":
            true_ids.append({"id": id, "claim": claim, "source": source})
        else:
            false_ids.append({"id": id, "claim": claim, "source": source})
    

    total_claims = len(results)
    claim_precision = len(true_ids) / total_claims
    claim_precision = round(claim_precision,3)
    
    total_citations = len(data["ref_abstract"]["cite_N"])
    citation_precision = len(true_ids) / total_citations
    citation_precision = round(citation_precision,3)
    
    total_sentences = len(count_sentences(related_work))
    citation_density = total_citations / total_sentences
    citation_density = round(citation_density,3)
    
    avg_citation_per_sentence = total_citations / total_claims
    avg_citation_per_sentence = round(avg_citation_per_sentence,3)
    
    
    direct_contradiction = 0
    information_not_present = 0
    misrepresentation = 0
    incorrect_attribution = 0
    other = 0
    
    for id in false_ids:
        source = id["source"]
        claim = id["claim"]
        prompt = process_data_for_classify_errors(claim,source)
        result = llm_client.get_completion(prompt)
        result = json_repair.loads(result)
        if "Direct Contradiction" in result["error_type"]:
            direct_contradiction += 1
        elif "Information Not Present / Unsubstantiated" in result["error_type"]:
            information_not_present += 1
        elif "Misrepresentation / Imprecise Wording" in result["error_type"]:
            misrepresentation += 1
        elif "Incorrect Attribution" in result["error_type"]:
            incorrect_attribution += 1
        else:
            other += 1
        errors_count = {
            "direct_contradiction": direct_contradiction,
            "information_not_present": information_not_present,
            "misrepresentation": misrepresentation,
            "incorrect_attribution": incorrect_attribution,
            "other": other
        }
    print(f"errors_count: {errors_count}")
    
    return {
        "claim_precision": claim_precision,
        "citation_precision": citation_precision,
        "citation_density": citation_density,
        "avg_citation_per_sentence": avg_citation_per_sentence,
        "direct_contradiction": direct_contradiction,
        "information_not_present": information_not_present,
        "misrepresentation": misrepresentation,
        "incorrect_attribution": incorrect_attribution,
        "other": other
    }
    
def main_eval_autosurvey():
    path = r"D:\Mydesktop\CitAgent\Code\Supplementary material and code\Multi_Agents_for_Citation_Verification\results\SurveyForge\multi-step reasoning methods for Large Language Models matches sentences and citations_100.json"
    with open(path, "r",encoding="utf-8") as f:
        data = json.load(f)
    related_work = """# Multi-Step Reasoning in Large Language Models: Methods, Challenges, and Future Directions

## 1 Introduction

Multi-step reasoning in Large Language Models (LLMs) represents a pivotal advancement in artificial intelligence, enabling models to tackle complex problems by decomposing them into intermediate steps. Unlike single-step inference, which relies on immediate responses, multi-step reasoning requires iterative and coherent processing of intermediate rationales to arrive at a final solution. This capability has been increasingly recognized as critical for tasks involving arithmetic, logical deduction, and commonsense reasoning, where direct inference often falls short. Recent studies, such as [1], have demonstrated that prompting LLMs to generate intermediate reasoning steps significantly enhances their performance on such tasks, achieving state-of-the-art results on benchmarks like GSM8K and MultiArith.  

Theoretical foundations of multi-step reasoning in LLMs are rooted in their ability to simulate sequential cognitive processes. These processes mirror human problem-solving strategies, where each step builds upon the previous one to refine the solution. Recent research [2] has shown that even without explicit training, LLMs can exhibit reasoning capabilities when prompted with simple cues like "Let's think step by step." This emergent behavior suggests that multi-step reasoning is an intrinsic property of sufficiently large models, enabled by their vast pretraining on diverse corpora. However, the extent to which these capabilities are systematic or merely dependent on surface-level patterns remains a subject of debate [3].  

One of the key innovations in this domain is the Chain-of-Thought (CoT) prompting method, which explicitly guides LLMs to generate step-by-step rationales before reaching a final answer. Empirical evidence [1] indicates that CoT significantly improves model performance on tasks requiring complex reasoning, such as mathematical problem-solving and symbolic logic. Variants of CoT, such as Self-Consistency and Iterative CoT, further enhance robustness by aggregating multiple reasoning paths or refining intermediate steps through iterative processes [4]. These advancements highlight the flexibility of LLMs in adapting to diverse reasoning paradigms.  

Despite these successes, multi-step reasoning in LLMs is not without challenges. Error propagation, where inaccuracies in early steps compound in later stages, remains a significant limitation. Additionally, models often struggle with tasks requiring precise logical consistency, such as deductive reasoning [5]. Recent approaches have sought to integrate symbolic reasoning modules with neural architectures to address these limitations, leveraging the strengths of both paradigms [6]. These hybrid methods demonstrate improved performance on tasks demanding strict logical rigor, such as theorem proving and constraint optimization.  

Another critical challenge is the computational cost associated with multi-step reasoning. Generating and verifying intermediate steps requires significant resources, limiting the scalability of these methods in real-world applications. Recent techniques, such as Early-Stopping Self-Consistency [7], aim to mitigate this issue by dynamically balancing inference efficiency with reasoning accuracy. These approaches highlight the ongoing trade-offs between performance and resource utilization in the design of multi-step reasoning systems.  

Emerging trends in multi-step reasoning focus on enhancing the interpretability and robustness of LLMs. Methods like Symbolic Chain-of-Thought [8] aim to ground reasoning processes in formal logic, providing verifiable and transparent rationales. Additionally, the integration of external tools and knowledge sources, such as retrieval-augmented generation and symbolic solvers, has shown promise in improving factual accuracy and reducing hallucinations [9]. These developments underscore the importance of hybrid approaches that combine neural and symbolic reasoning for more reliable and scalable solutions.  

Looking ahead, future research directions in multi-step reasoning include the development of self-improving frameworks that enable LLMs to iteratively refine their reasoning capabilities through feedback mechanisms. Techniques like Step-wise Preference Optimization [10] leverage fine-grained supervision to enhance the correctness of intermediate steps, addressing the limitations of coarse-grained reward models. Furthermore, the exploration of multimodal reasoning, where LLMs integrate textual, visual, and structured data, represents a promising frontier for extending multi-step reasoning to more diverse and complex domains [11].  

In summary, multi-step reasoning in LLMs has emerged as a transformative capability, enabling models to tackle complex problems through iterative and coherent rationales. While significant progress has been made, challenges related to error propagation, logical consistency, and computational efficiency persist. The integration of symbolic reasoning, external tools, and multimodal data offers promising avenues for addressing these limitations and advancing the field. As research continues to bridge the gap between human-like reasoning and machine capabilities, multi-step reasoning will remain a cornerstone of progress in artificial intelligence.

## 2 Foundational Techniques for Multi-Step Reasoning

### 2.1 Chain-of-Thought Prompting and Variants

Here is the corrected subsection with verified citations:

Chain-of-Thought (CoT) prompting has emerged as a pivotal technique for enhancing the multi-step reasoning capabilities of large language models (LLMs) by decomposing complex problems into intermediate reasoning steps. This approach, introduced by [1], leverages the implicit knowledge of LLMs to generate step-by-step solutions, significantly improving performance on tasks requiring arithmetic, commonsense, and symbolic reasoning. The core premise of CoT lies in its ability to mimic human-like deliberation, where intermediate steps serve as a scaffold for deriving final answers. Empirical studies demonstrate that CoT prompting can elevate the accuracy of a 540B-parameter model on math benchmarks like GSM8K by over 40 percentage points compared to direct prompting [1].

The efficacy of CoT hinges on the quality and structure of the reasoning chains. [2] reveal that even simple zero-shot prompts like "Let’s think step by step" can elicit effective reasoning, suggesting that LLMs possess latent step-by-step reasoning capabilities. However, the robustness of these chains varies with task complexity. For instance, CoT struggles with proof planning when multiple valid intermediate steps exist, as models often fail to systematically explore alternative paths [12]. This limitation underscores the necessity for controlled prompting strategies, such as [13], which prioritizes demonstrations with higher-step reasoning to improve generalization. 

Recent advances have introduced self-consistency mechanisms to address error propagation in CoT. By sampling multiple reasoning paths and aggregating answers via majority voting, self-consistency mitigates inconsistencies in individual chains [1]. Further refinements, such as [14], dynamically correct errors during reasoning by iteratively revising intermediate steps. However, these methods increase computational costs, prompting research into efficiency optimizations. For example, [7] reduces sampling overhead by 80% on tasks like GSM8K while maintaining accuracy.

A critical challenge for CoT is its susceptibility to hallucination and logical inconsistencies. Studies show that models often generate plausible but incorrect steps, particularly in abstract reasoning domains [3]. To enhance faithfulness, techniques like [15] decompose verification into independent subprocesses, reducing hallucination by 20–30% on fact-based tasks. Similarly, [16] ensures step-wise validity by translating natural language chains into symbolic proofs. These approaches highlight a growing trend toward hybrid neuro-symbolic frameworks, where LLMs interface with formal solvers to enforce logical rigor [5].

Emerging variants extend CoT beyond textual reasoning. [11] integrates visual and textual modalities, achieving a 16-point accuracy gain on ScienceQA by separating rationale generation and answer inference. Meanwhile, neuro-symbolic architectures like [17] employ SQL databases as external memory to track facts during multi-step reasoning. Such innovations underscore the versatility of CoT paradigms across domains, though they also reveal scalability trade-offs. For instance, [18] caches reusable thought templates to improve efficiency, reducing inference costs by 88% while maintaining performance.

Theoretical analyses of CoT reveal its parallels with computational models. [19] prove that CoT enables constant-depth transformers to solve problems in P by simulating serial computation, a capability absent in standard prompting. However, the interplay between CoT and model architecture remains underexplored. Mechanistic studies suggest that middle transformer layers prioritize pretraining knowledge, while later layers focus on in-context reasoning [20], implying that optimal CoT performance requires architectural alignment.

Future directions for CoT include dynamic prompt optimization and cross-modal generalization. Approaches like [21] adaptively select reasoning strategies per task, while frameworks like [22] employ Monte Carlo Tree Search to explore reasoning spaces. However, fundamental limitations persist, such as the inability to eliminate hallucinations due to inherent undecidability in LLMs [23]. Addressing these challenges will require advances in verifiable reasoning and scalable supervision, positioning CoT as a fertile ground for innovation in reliable AI systems.

### 2.2 Integration of Symbolic and Neural Reasoning

The integration of symbolic and neural reasoning represents a pivotal frontier in enhancing the multi-step reasoning capabilities of Large Language Models (LLMs). While neural approaches excel at pattern recognition and generalization, they often struggle with tasks requiring precise, rule-based reasoning, such as mathematical proofs or logical deductions. Conversely, symbolic methods offer rigorous computational structures but lack the flexibility to handle ambiguous or incomplete information. Hybrid approaches aim to bridge these gaps, leveraging the complementary strengths of both paradigms to achieve more robust and interpretable reasoning systems.

One promising direction is the development of neurosymbolic architectures, where symbolic modules are embedded within neural networks to support precise rule-based reasoning. For instance, [8] introduces SymbCoT, a framework that translates natural language contexts into symbolic formats and employs logical rules to guide step-by-step reasoning. SymbCoT not only improves the faithfulness of reasoning chains but also enhances the explainability of LLM outputs by explicitly aligning intermediate steps with formal logic. Similarly, [24] proposes the Graph of Thought (GoT), which organizes reasoning steps into a structured graph, allowing LLMs to navigate complex dependencies and optimize reasoning pathways through symbolic abstractions.

Another key innovation is the use of symbolic working memory to augment neural reasoning. Techniques such as [10] introduce external memory mechanisms that track facts and rules during multi-step deductive tasks. These memory-augmented systems enable LLMs to maintain consistency across reasoning steps, reducing error propagation and improving task performance. For example, [18] employs a meta-buffer to store high-level thought templates, which are adaptively instantiated for specific tasks. This approach not only enhances reasoning accuracy but also improves efficiency by reusing proven reasoning structures.

Inductive logic learning represents another critical area of integration. Methods like [25] use LLMs to generate and verify logical rules from training examples, creating a rule library that can be applied during deduction. This approach allows LLMs to perform systematic reasoning without task-specific supervision, making them more adaptable to diverse problem domains. Furthermore, [26] extends this idea by formulating CoT reasoning into a structured multi-round QA format, where LLMs interact with external knowledge bases to verify and modify reasoning traces. This interaction ensures that intermediate steps are grounded in verified facts, addressing the issue of hallucination and error propagation.

Despite these advancements, several challenges remain. The integration of symbolic and neural methods often introduces computational overhead, particularly in tasks requiring real-time reasoning. Moreover, the interpretability of hybrid systems can be compromised when symbolic rules are embedded within opaque neural architectures. Future research should focus on developing more efficient mechanisms for symbolic-neural interaction, as well as methods for ensuring the transparency and consistency of reasoning chains. Additionally, there is a need for standardized benchmarks and evaluation metrics to assess the performance of hybrid approaches across different reasoning tasks.

In conclusion, the integration of symbolic and neural reasoning offers a powerful framework for enhancing the multi-step reasoning capabilities of LLMs. By combining the flexibility of neural networks with the rigor of symbolic systems, these approaches address key limitations in pure neural reasoning, paving the way for more reliable and interpretable AI systems. As the field continues to evolve, further innovations in neurosymbolic architectures, memory augmentation, and inductive logic learning will be critical to unlocking the full potential of LLMs in complex reasoning tasks.

### 2.3 Role of Attention and Memory Mechanisms

The capacity of large language models (LLMs) to perform multi-step reasoning fundamentally relies on their ability to manage and propagate information across sequential processing steps. Central to this capability are attention and memory mechanisms, which orchestrate the flow of context and intermediate states to maintain logical coherence. Recent research has demonstrated that transformer-based architectures intrinsically encode reasoning pathways through dynamic attention patterns, with specific attention heads specializing in information routing for tasks requiring multi-hop inference [27]. However, vanilla attention distributions often suffer from skewed focus, disproportionately weighting local tokens over distant but critical dependencies. This limitation manifests in error propagation during complex reasoning chains, where early misattentions cascade into incorrect conclusions [3]. To mitigate this, novel techniques like rebalancing attention scores via learned gating mechanisms have been proposed, enabling models to dynamically prioritize semantically relevant tokens across reasoning steps [28].

Memory architectures further augment reasoning by providing explicit storage for intermediate computations. Early approaches integrated differentiable memory buffers to cache partial results, as seen in [17], where SQL-like structured memory allows precise retrieval of facts during deduction. More sophisticated frameworks, such as RecallM’s updatable memory, inject task-specific biases into attention heads through learned memory slots, enabling models to perform multi-hop inference by incrementally refining representations [1]. These mechanisms parallel human working memory, where temporary storage facilitates iterative hypothesis testing—a phenomenon empirically validated in [29], which shows that models trained on locally clustered variables exhibit stronger compositional reasoning when equipped with memory-augmented attention. The interplay between memory and attention is formalized in residual connection-based frameworks like RESPROMPT, where graph-inspired prompts create non-linear pathways for cross-step information flow, demonstrated to improve performance on deductive tasks by 18% [24].

A critical advancement lies in hybrid neuro-symbolic memory systems, which marry neural flexibility with symbolic precision. [5] introduces modular memory units that interface with external theorem provers, where attention heads selectively activate symbolic rules during inference. This approach reduces hallucination by 39% on FOLIO compared to pure neural baselines, as symbolic constraints guide attention toward logically valid paths. Similarly, [18] proposes meta-buffers storing reusable reasoning templates, enabling models to dynamically retrieve and adapt proven strategies—akin to algorithmic phase transitions observed in transformer layers during problem-solving [20]. Such architectures address the brittleness of implicit memory in pure transformers, where latent representations often fail to sustain long-range dependencies [19].

Emerging findings reveal that optimal reasoning performance requires balancing static and dynamic memory access. Static components (e.g., pretrained weights) encode inductive biases for logical operations, while dynamic modules (e.g., attention-based memory) adapt to task-specific contexts. The tension between these is evident in [30], where symbolic task decomposition improves mathematical reasoning fidelity but incurs computational overhead. Compromise solutions involve sparse memory access patterns, as in [31], which uses Monte Carlo Tree Search to prune low-probability memory retrievals during reasoning. Future directions may explore biologically inspired architectures, such as oscillatory attention mechanisms that rhythmically gate memory updates—a concept hinted at in [32], where phased retrieval boosts systematic generalization.

The frontier of this research hinges on overcoming three key challenges: (1) the quadratic cost of cross-step attention in long reasoning chains, (2) the semantic drift of memory representations during iterative updates, and (3) the integration of implicit and explicit memory systems for scalable reasoning. Solutions may lie in latent space disentanglement techniques or resource-efficient architectures like [33], which couples hypothesis generation with memory validation loops. As emphasized in [34], the ultimate goal is achieving human-like reliability in multi-step inference—a feat requiring synergistic advances in both architectural design and training paradigms for attention and memory systems.

### 2.4 Self-Verification and Iterative Refinement

Here is the subsection with corrected citations:

Self-verification and iterative refinement represent a paradigm shift in multi-step reasoning for large language models (LLMs), addressing critical limitations in error propagation and factual consistency. These techniques enable LLMs to autonomously validate intermediate reasoning steps and dynamically optimize solution paths, bridging the gap between single-pass generation and robust logical reasoning. The efficacy of such methods derives from their dual focus on internal consistency checks and external feedback integration, as demonstrated by frameworks like LLM-ARC and Self-Discover [35].

At the core of self-verification lies backward validation, where LLMs assess the logical soundness of generated reasoning chains through inverse reasoning. For instance, MathPrompter [36] employs symbolic solvers to verify the correctness of each step, filtering invalid inferences before final answer generation. This approach mirrors human problem-solving heuristics, where intermediate conclusions are tested against first principles. Empirical studies reveal a 15-25% accuracy improvement on mathematical reasoning tasks when such validation is applied [36]. However, the computational overhead of symbolic verification remains a key trade-off, particularly for real-time applications.

Iterative refinement extends this concept through dynamic context adaptation, exemplified by inner dialogue frameworks like IoT [14]. These systems establish feedback loops where initial reasoning traces are critiqued and revised based on self-generated or external evaluations. The Plan-and-Solve (PS) prompting method [37] operationalizes this through a two-phase process: task decomposition followed by stepwise execution with error correction. When enhanced with detailed instructions (PS+), this approach reduces missing-step errors by 32% compared to standard Chain-of-Thought prompting [37].

A critical advancement in this domain is meta-reasoning prompting (MRP) [21], which dynamically selects reasoning strategies based on task characteristics. By evaluating multiple reasoning chains and their intermediate states, MRP identifies optimal inference paths while pruning contradictory trajectories. This method demonstrates particular efficacy in multi-hop QA, outperforming majority-voting baselines by 7-12% across diverse benchmarks [21]. Theoretically, such approaches align with Bayesian inference principles, where the LLM acts as a hierarchical estimator aggregating evidence from parallel reasoning pathways.

Comparative analysis reveals distinct trade-offs among these methods. Self-verification techniques excel in precision but suffer from computational intensity [38], whereas iterative methods like Progressive-Hint Prompting (PHP) [39] prioritize efficiency through sequential hint generation. Hybrid approaches, such as the Buffer of Thoughts (BoT) framework [18], attempt to reconcile these trade-offs by maintaining reusable thought templates that undergo continuous refinement.

Emerging challenges include the "confirmation bias paradox" identified in retrieval-augmented systems [40], where LLMs disproportionately favor parametric knowledge over conflicting external evidence. Additionally, the impact of reasoning step length on verification efficacy remains underexplored – longer chains exhibit higher error accumulation despite improved task decomposition [28]. Future directions point toward neuro-symbolic architectures that combine learned verification heuristics with formal proof systems [17], and the integration of explicit memory modules to support cross-episodic error correction [41]. The development of benchmarks measuring reasoning faithfulness, as opposed to mere answer accuracy [42], will be critical for advancing this field.

### 2.5 Emergent Techniques and Modular Reasoning

Recent advances in multi-step reasoning have shifted toward modular and graph-structured paradigms, which address scalability and compositional generalization challenges inherent in linear Chain-of-Thought (CoT) approaches. These emergent techniques decompose reasoning into specialized sub-tasks or leverage non-linear topological representations to improve precision, efficiency, and interpretability.  

A prominent direction involves *Graph of Thoughts (GoT)* frameworks, which model reasoning steps as nodes in a directed acyclic graph (DAG), allowing dynamic path exploration and synthesis of intermediate conclusions. For instance, [18] introduces a meta-buffer to store reusable "thought-templates" distilled from diverse tasks, which are instantiated adaptively for new problems. This method reduces redundancy in intermediate computations, achieving up to 51% accuracy gains on Checkmate-in-One while consuming only 12% of the compute cost of tree-based methods. Similarly, [6] integrates symbolic solvers to guide LLM-generated steps, ensuring logical validity by recursively decomposing queries into sub-goals. The framework's top-down control mechanism demonstrates 20% improvement on ProofWriter over traditional backward chaining, highlighting the efficacy of hybrid neuro-symbolic architectures for faithful reasoning.  

Modularization strategies further enhance scalability by partitioning reasoning into task-specific units. [43] formalizes problems as theorems in Lean, leveraging its proof library to verify step correctness, achieving near-state-of-the-art performance on FOLIO with fewer than 100 in-domain samples. This aligns with findings in [5], where symbolic modules correct LLM outputs via error messages, improving accuracy by 39.2% on logical datasets. Such decomposition also underpins [10], which fine-tunes models using preference-ranked reasoning paths from tree-search processes, closing the performance gap between CoT and cost-intensive Tree-of-Thought (ToT) methods.  

Key trade-offs emerge between flexibility and control in these approaches. Graph-based methods like [44] enable iterative refinement through DAG-structured critiques but face higher memory overheads. Modular systems, as in [35], autonomously select and chain atomic reasoning modules (e.g., critical thinking, step-by-step inference), yet require careful calibration to avoid error propagation across sub-tasks. The "complexity sweet spot" identified in [28] suggests that longer reasoning chains improve performance only when aligned with task demands, underscoring the need for adaptive modularity.  

Future directions should address two critical challenges: (1) *Dynamic Compositionality*, where modules or graph nodes are assembled contextually, as proposed in [45]; and (2) *Verification Scalability*, ensuring efficient validation of reasoning steps without relying on external solvers. The integration of retrieval-augmented modules, as explored in [18], and latent-space reasoning analysis, as in [46], offers promising avenues. Empirical results from [47] further suggest that combining reward-guided search with lightweight world models could unify these paradigms, enabling both robustness and efficiency in multi-step reasoning.

## 3 Advanced Reasoning Frameworks and Architectures

### 3.1 Tree-Based and Graph-Based Reasoning Approaches

Tree-based and graph-based reasoning approaches represent a paradigm shift in enhancing the multi-step reasoning capabilities of large language models (LLMs) by providing structured frameworks for exploring diverse reasoning paths and their combinations. These methods depart from linear Chain-of-Thought (CoT) techniques by introducing hierarchical or relational structures that enable more systematic and efficient problem-solving. Tree-based frameworks, such as the Tree-of-Thought (ToT) approach, decompose problems into hierarchical reasoning steps, allowing for parallel exploration of multiple solution paths and dynamic pruning of incorrect branches. This architecture not only mitigates error propagation but also aligns closer to human reasoning, where hypotheses are generated, evaluated, and refined iteratively. Empirical studies have demonstrated that ToT significantly outperforms traditional CoT in complex tasks such as mathematical reasoning and strategic planning [3].

Graph-based reasoning, on the other hand, leverages relational representations to facilitate multi-hop inference and path optimization. By encoding knowledge and intermediate reasoning steps as nodes and edges in a graph, these methods enable LLMs to navigate complex problem spaces more effectively. For instance, the Graph of Thoughts (GoT) framework outperforms linear CoT in tasks requiring spatial and relational reasoning by constructing reasoning topologies that capture non-linear dependencies between steps [18]. This approach is particularly advantageous in scenarios where reasoning involves traversing multiple entities and relationships, as it allows for parallel exploration and aggregation of diverse reasoning pathways.

A critical strength of tree-based and graph-based approaches lies in their ability to integrate heuristic-guided search algorithms, such as Monte Carlo Tree Search (MCTS), to prioritize high-probability reasoning paths during multi-step inference [22]. However, these methods also face limitations, including increased computational complexity and the challenge of effectively balancing exploration and exploitation in large reasoning spaces. Recent innovations, such as the Buffer of Thoughts (BoT) framework, aim to address these issues by storing reusable thought-templates in a meta-buffer, reducing redundant computations while maintaining reasoning robustness across diverse tasks [18].

Emerging trends in this domain focus on hybrid approaches that combine the strengths of tree-based decomposition with graph-based relational reasoning. For example, the K-Level Reasoning framework introduces recursive k-level thinking to address dynamic reasoning challenges, enabling LLMs to anticipate and adapt to evolving problem contexts [48]. Additionally, the integration of symbolic solvers with graph-based frameworks, as seen in SymBa (Symbolic Backward Chaining), enhances the interpretability and faithfulness of reasoning processes by grounding them in formal logic [6].

Despite these advancements, key challenges remain, including the need for scalable evaluation metrics to assess the quality of reasoning paths and the trade-offs between computational efficiency and reasoning depth. Future directions suggest the potential for combining tree-based and graph-based approaches with reinforcement learning to develop autonomous reasoning systems capable of iterative self-improvement [49]. Furthermore, the integration of multimodal data sources into these frameworks could unlock new avenues for complex reasoning in sensory-rich environments, paving the way for more human-like AI systems.

### 3.2 Modular Architectures for Specialized Reasoning

Modular architectures represent a paradigm shift in enhancing the reasoning capabilities of large language models (LLMs), addressing the limitations of monolithic models in handling complex, multi-step tasks. By decomposing reasoning processes into specialized sub-networks or collaborative modules, these architectures improve both precision and scalability. A prominent example is the Decomposition-solver framework (DaSLaM), which employs smaller LMs to break down problems into solvable sub-tasks before delegating them to larger LLMs, thereby reducing computational overhead and mitigating error propagation [50]. This approach demonstrates that strategic task partitioning can significantly enhance performance, particularly in mathematical and symbolic reasoning domains where error accumulation in long chains is a critical challenge.

Neuro-symbolic hybrid modules further exemplify this trend, combining the flexibility of neural networks with the rigor of symbolic solvers. For instance, Logic-LM integrates formal logic engines to validate intermediate reasoning steps, ensuring deductive correctness while retaining the generative capacity of LLMs [8]. Such architectures address the hallucination problem by grounding neural outputs in verifiable symbolic operations, achieving state-of-the-art results on constrained reasoning tasks. The SymbCoT framework extends this by translating natural language reasoning chains into symbolic expressions, enabling stepwise verification and error correction [8]. These hybrid systems reveal a fundamental trade-off: while symbolic components enhance faithfulness, their integration often requires task-specific engineering, limiting generalizability.

Multi-agent reasoning systems represent another innovative direction, where LLMs function as specialized agents (e.g., validators, critics) to collaboratively refine reasoning chains. The PRER framework decomposes legal and financial queries into Planner-Reasoner-Executor-Reflector modules, each addressing distinct subtasks while maintaining interpretability [24]. Similarly, ART leverages a library of task-specific demonstrations to automatically generate modular reasoning programs, dynamically pausing execution to incorporate external tool outputs [9]. These systems showcase how modularity enables adaptive reasoning but face challenges in coordinating inter-module communication and avoiding cascading errors.

Emerging work explores the potential of meta-architectures to orchestrate modular reasoning. The Buffer of Thoughts (BoT) approach introduces a meta-buffer storing reusable thought-templates distilled from diverse problem-solving processes, allowing LLMs to adaptively instantiate these templates for new tasks [18]. This method achieves notable efficiency gains, requiring only 12% of the computational cost of traditional multi-query methods while maintaining competitive performance. However, its effectiveness depends heavily on the representational breadth of the meta-buffer, raising questions about scalability to open-domain problems.

Theoretical analyses suggest that modular architectures align with the intrinsic compositional structure of reasoning tasks. The Hypotheses-to-Theories (HtT) framework formalizes this by distilling task-specific rules into a rule library during training, which is then deployed deductively during inference [25]. This two-stage process—induction followed by deduction—mirrors human reasoning patterns and significantly outperforms end-to-end approaches on relational and numerical reasoning tasks. Nevertheless, rule induction remains computationally intensive, and the quality of learned rules varies substantially with task complexity.

Key challenges persist in designing modular systems. First, dynamic routing between modules—deciding which sub-network should handle a given reasoning step—lacks robust theoretical foundations, often relying on heuristic-based selection. Second, the trade-off between module specialization and cross-task transferability remains understudied, with overly specialized modules risking brittle performance on distribution shifts. Third, while neuro-symbolic hybrids excel in deterministic domains, their application to probabilistic or open-ended reasoning tasks warrants further exploration.

Future directions may focus on self-organizing modular architectures, where the decomposition and allocation of reasoning subtasks emerge dynamically from problem constraints. Integrating reinforcement learning to optimize module selection, as proposed in Chain of Preference Optimization (CPO), could further enhance adaptability [10]. Additionally, unifying modular approaches with geometric reasoning frameworks—such as representing reasoning states in latent spaces—may offer new pathways for scalable multi-step inference. As the field progresses, the interplay between architectural innovation and theoretical grounding will be critical to advancing the frontiers of specialized reasoning in LLMs.

### 3.3 Self-Verification and Reflection Mechanisms

Self-verification and reflection mechanisms have emerged as critical components in enhancing the reasoning capabilities of large language models (LLMs), addressing issues such as hallucination, logical inconsistencies, and error propagation in multi-step reasoning tasks. These methods enable LLMs to autonomously critique and refine their outputs, mimicking human-like iterative reasoning processes. Key approaches include automated self-critique frameworks, stepwise preference optimization, and dynamic reflection prompting, each offering distinct advantages and limitations.

Automated self-critique frameworks, such as those employed in [51], leverage symbolic solvers to validate intermediate reasoning steps. These frameworks generate tests for each reasoning step and use external verifiers to filter incorrect conclusions, significantly improving logical consistency. For example, [5] integrates symbolic solvers to refine LLM-generated reasoning chains, achieving a 39.2% performance boost over standard prompting. However, these methods often rely on external symbolic systems, which can introduce computational overhead and require precise translation between natural language and formal logic.

Stepwise preference optimization, as explored in [29], uses error traces from failed reasoning paths to train models to avoid recurring mistakes. By analyzing patterns in incorrect intermediate steps, LLMs learn to prioritize reliable reasoning paths. This approach is particularly effective in tasks requiring multiple inference steps, as demonstrated in [30], where LLMs improved their ability to handle symbolic tasks by incorporating feedback from error analysis. However, the efficacy of stepwise optimization depends heavily on the quality and diversity of the error traces, limiting its scalability in complex reasoning scenarios.

Dynamic reflection prompting, such as Self-Contrast, encourages LLMs to compare divergent reasoning perspectives to identify and resolve inconsistencies. By generating multiple reasoning paths and contrasting their outcomes, LLMs can iteratively refine their conclusions. This method is particularly effective in open-ended reasoning tasks, as it allows models to explore alternative solutions and select the most coherent one. However, dynamic reflection can be computationally intensive, requiring multiple iterations of generation and comparison.

Emerging trends in self-verification and reflection include hybrid neuro-symbolic approaches, as seen in [52], which combine LLMs with backward-chaining solvers to improve logical validity. Additionally, frameworks like Buffer of Thoughts (BoT) [18] introduce meta-buffers to store high-level reasoning templates, enabling models to adaptively instantiate these templates for efficient problem-solving. BoT demonstrates significant improvements in reasoning tasks, achieving up to a 51% boost in accuracy on complex benchmarks while reducing inference costs.

Despite these advancements, several challenges remain. The reliance on symbolic solvers for verification introduces integration complexities, and the computational demands of dynamic reflection prompting limit its practicality. Furthermore, the effectiveness of self-critique frameworks is constrained by the LLMs' ability to accurately identify and correct their own errors. Future directions include developing hybrid frameworks that combine self-verification with external knowledge sources, such as retrieval-augmented generation, and exploring reinforcement learning techniques to optimize reflection processes. These advancements promise to enhance the robustness and reliability of LLMs in complex reasoning tasks, paving the way for more trustworthy and interpretable AI systems.

### 3.4 Attention and Geometric Foundations of Reasoning

The attention mechanism serves as the computational backbone for multi-step reasoning in large language models (LLMs), dynamically orchestrating information flow across reasoning steps. Recent mechanistic studies reveal that attention heads encode hierarchical reasoning pathways, where early layers resolve latent intermediate variables (e.g., bridge entities in multi-hop queries) before later layers perform compositional operations [53]. """
    result_eval = eval_autosurvey(data,related_work)
    print(result_eval)
    

def main_eval_fast_macg():
    with open(r"D:\Mydesktop\CitAgent\Code\Supplementary material and code\results\2406_02541\results_20_2_0.json", "r",encoding="utf-8") as f:
        result = json.load(f)
    related_work = result["related_work"]
    cite_ids = result["citations"]
    quotes_with_citation_info = result["quotes_with_citation_info"]
    result_eval = eval_citation_fast_macg(related_work,cite_ids, quotes_with_citation_info)
    print(result_eval)
        

def main_eval():
    with open(r"D:\Mydesktop\CitAgent\Code\Supplementary material and code\Multi_Agents_for_Citation_Verification\results\MACG\result_MACG_test.json", "r",encoding="utf-8") as f:
        result = json.load(f)
    claim_precision_list = []
    citation_precision_list = []
    citation_density_list = []
    avg_citation_per_sentence_list = []
    direct_contradiction_list = []
    information_not_present_list = []
    misrepresentation_list = []
    incorrect_attribution_list = []
    other_list = []
    for i in range(len(result)):
        related_work = result[i]["related_work"]["related_works"]
        cite_ids = result[i]["related_work"]["cite_ids"]
        selected_papers = result[i]["related_work"]["selected_papers"]
        result_eval = eval_citation(related_work,cite_ids, selected_papers)
    
        claim_precision = result_eval["claim_precision"]
        citation_precision = result_eval["citation_precision"]
        citation_density = result_eval["citation_density"]
        avg_citation_per_sentence = result_eval["avg_citation_per_sentence"]
        direct_contradiction = result_eval["direct_contradiction"]
        information_not_present = result_eval["information_not_present"]
        misrepresentation = result_eval["misrepresentation"]
        incorrect_attribution = result_eval["incorrect_attribution"]
        other = result_eval["other"]
        
        claim_precision_list.append(claim_precision)
        citation_precision_list.append(citation_precision)
        citation_density_list.append(citation_density)
        avg_citation_per_sentence_list.append(avg_citation_per_sentence)
        direct_contradiction_list.append(direct_contradiction)
        information_not_present_list.append(information_not_present)
        misrepresentation_list.append(misrepresentation)
        incorrect_attribution_list.append(incorrect_attribution)
        other_list.append(other)
    print(f"Claim Precision: {sum(claim_precision_list) / len(claim_precision_list)}")
    print(f"Citation Precision: {sum(citation_precision_list) / len(citation_precision_list)}")
    print(f"Citation Density: {sum(citation_density_list) / len(citation_density_list)}")
    print(f"Avg Citation Per Sentence: {sum(avg_citation_per_sentence_list) / len(avg_citation_per_sentence_list)}")
    print(f"Direct Contradiction: {sum(direct_contradiction_list)}")
    print(f"Information Not Present: {sum(information_not_present_list)}")
    print(f"Misrepresentation: {sum(misrepresentation_list)}")
    print(f"Incorrect Attribution: {sum(incorrect_attribution_list)}")
    print(f"Other: {sum(other_list)}")
    
    
    return result
    
    

    
    
def main_eval_litllm():
    with open(r"D:\Mydesktop\CitAgent\Code\Supplementary material and code\Multi_Agents_for_Citation_Verification\results\litllm\d_0_results\RAG_generated_related_work.json", "r",encoding="utf-8") as f:
        result = json.load(f)
    dir_path = r"D:\Mydesktop\CitAgent\Code\Supplementary material and code\Multi_Agents_for_Citation_Verification\results\litllm\d_0"
    
    claim_precision_list = []
    citation_precision_list = []
    citation_density_list = []
    avg_citation_per_sentence_list = []
    direct_contradiction_list = []
    information_not_present_list = []
    misrepresentation_list = []
    incorrect_attribution_list = []
    other_list = []
    
    data_list = []
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        with open(file_path, "r",encoding="utf-8") as f:
            data = json.load(f)
            data_list.append(data)
    
    predictions = result["predictions"]
    related_work_list = []
    for prediction in predictions:
        related_work_list.append(prediction)
        
    for data,related_work in zip(data_list,related_work_list):
        result = eval_litllm(data,related_work)
        claim_precision = result["claim_precision"]
        citation_precision = result["citation_precision"]
        citation_density = result["citation_density"]
        avg_citation_per_sentence = result["avg_citation_per_sentence"]
        direct_contradiction = result["direct_contradiction"]
        information_not_present = result["information_not_present"]
        misrepresentation = result["misrepresentation"]
        incorrect_attribution = result["incorrect_attribution"]
        other = result["other"]
        
        claim_precision_list.append(claim_precision)
        citation_precision_list.append(citation_precision)
        citation_density_list.append(citation_density)
        avg_citation_per_sentence_list.append(avg_citation_per_sentence)
        direct_contradiction_list.append(direct_contradiction)
        information_not_present_list.append(information_not_present)
        misrepresentation_list.append(misrepresentation)
        incorrect_attribution_list.append(incorrect_attribution)
        other_list.append(other)
    print(f"Claim Precision: {sum(claim_precision_list) / len(claim_precision_list)}")
    print(f"Citation Precision: {sum(citation_precision_list) / len(citation_precision_list)}")
    print(f"Citation Density: {sum(citation_density_list) / len(citation_density_list)}")
    print(f"Avg Citation Per Sentence: {sum(avg_citation_per_sentence_list) / len(avg_citation_per_sentence_list)}")
    print(f"Direct Contradiction: {sum(direct_contradiction_list)}")
    print(f"Information Not Present: {sum(information_not_present_list)}")
    print(f"Misrepresentation: {sum(misrepresentation_list)}")
    print(f"Incorrect Attribution: {sum(incorrect_attribution_list)}")
    print(f"Other: {sum(other_list)}")

    return result
    
    




if __name__ == "__main__":
    # main_eval()
    # main_eval_litllm()
    # main_eval_fast_macg()
    main_eval_autosurvey()