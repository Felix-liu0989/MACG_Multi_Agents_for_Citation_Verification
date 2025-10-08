"""
Deep Research Agents评估系统使用示例
展示如何使用基于MACG框架的评估系统来评测deep research agents
"""

import json
import os
from evaluation.deep_research_evaluation_pipeline import DeepResearchEvaluationPipeline, EvaluationConfig
from evaluation.deep_research_evaluator import DeepResearchEvaluator
from evaluation.baseline_comparison import BaselineComparisonEvaluator

def example_single_evaluation():
    """单个摘要评估示例"""
    print("=== 单个摘要评估示例 ===")
    
    # 创建配置
    config = EvaluationConfig(
        llm_provider="gemini",
        breadth=10,
        depth=2,
        diversity=0.0,
        enable_baseline_comparison=True,
        enable_detailed_analysis=True
    )
    
    # 创建评估流水线
    pipeline = DeepResearchEvaluationPipeline(config)
    
    # 示例摘要
    abstract = """
    This study presents a novel human–AI collaborative framework for transitivity annotation within Systemic Functional Linguistics (SFL), focusing on seven high-frequency English verbs (find, make, allow, give, tell, go, take). Traditional manual annotation of Hallidayan process types—material, mental, verbal, relational (attributive/identifying), behavioral, and existential—is labor-intensive and limited by scalability. While large language models (LLMs) offer potential for automation, their application to SFL transitivity remains unexplored.
    We designed a five-round prompt-engineering workflow using DeepSeek-Chat, progressively incorporating expert knowledge through zero-shot, one-shot, verb-specific few-shot, error-explanation, and common pit-fall prompts. Each round evaluated model performance against a 3,000-clause gold standard annotated by SFL-trained experts, measured via accuracy, recall, macro-F1, and Cohen's κ. A Python pipeline was developed to diagnose model–human discrepancies, generate model self-explanations, and distill recurring errors into few-shot cues for subsequent rounds.
    """
    
    # 执行评估
    result = pipeline.evaluate_single_abstract(abstract, "example_paper_1")
    
    # 打印结果
    print(f"论文ID: {result['paper_id']}")
    print(f"评估时间: {result['timestamp']}")
    
    if result.get("evaluation_result"):
        eval_result = result["evaluation_result"]
        print(f"总体分数: {eval_result['overall_score']:.3f}")
        print(f"引用准确性: {eval_result['citation_reliability']['citation_accuracy']:.3f}")
        print(f"事实准确性: {eval_result['factual_support']['factual_accuracy']:.3f}")
        print(f"幻觉率: {eval_result['factual_support']['hallucination_rate']:.3f}")
    
    if result.get("recommendations"):
        print("\n改进建议:")
        for i, rec in enumerate(result["recommendations"], 1):
            print(f"{i}. {rec}")
    
    return result

def example_batch_evaluation():
    """批量评估示例"""
    print("\n=== 批量评估示例 ===")
    
    # 创建配置
    config = EvaluationConfig(
        llm_provider="gemini",
        breadth=8,
        depth=2,
        diversity=0.2,
        enable_baseline_comparison=True
    )
    
    # 创建评估流水线
    pipeline = DeepResearchEvaluationPipeline(config)
    
    # 示例摘要列表
    abstracts = [
        {
            "paper_id": "paper_1",
            "abstract": "This paper presents a novel approach to natural language processing using transformer architectures..."
        },
        {
            "paper_id": "paper_2", 
            "abstract": "We propose a new method for computer vision tasks based on convolutional neural networks..."
        },
        {
            "paper_id": "paper_3",
            "abstract": "This study investigates the application of machine learning in healthcare applications..."
        }
    ]
    
    # 执行批量评估
    results = pipeline.evaluate_batch(abstracts, "batch_evaluation_results.json")
    
    # 打印摘要统计
    print(f"处理了 {len(results)} 个摘要")
    
    successful_results = [r for r in results if r.get("evaluation_result")]
    print(f"成功评估: {len(successful_results)} 个")
    
    if successful_results:
        overall_scores = [r["evaluation_result"]["overall_score"] for r in successful_results]
        avg_score = sum(overall_scores) / len(overall_scores)
        print(f"平均总体分数: {avg_score:.3f}")
    
    return results

def example_baseline_comparison():
    """基线对比示例"""
    print("\n=== 基线对比示例 ===")
    
    # 创建基线对比评估器
    comparator = BaselineComparisonEvaluator(
        llm_provider="gemini"
    )
    
    # 示例摘要
    abstract = """
    This paper introduces a new framework for evaluating the reliability of citations in academic writing. 
    We propose a multi-dimensional approach that considers citation accuracy, relevance, timeliness, and completeness. 
    Our method uses large language models to automatically assess citation quality and provide feedback for improvement.
    """
    
    # 执行综合对比
    comparison_result = comparator.comprehensive_comparison(abstract)
    
    # 打印对比结果
    print("方法对比结果:")
    print(f"MACG总体分数: {comparison_result.macg_result.evaluation_result.overall_score:.3f}")
    print(f"Perplexity总体分数: {comparison_result.perplexity_result.evaluation_result.overall_score:.3f}")
    print(f"Naive RAG总体分数: {comparison_result.naive_rag_result.evaluation_result.overall_score:.3f}")
    print(f"GPT Search总体分数: {comparison_result.gpt_search_result.evaluation_result.overall_score:.3f}")
    
    # 打印排名
    rankings = comparison_result.comparison_summary["rankings"]["overall"]
    print(f"\n总体排名: {rankings}")
    
    return comparison_result

def example_custom_evaluation():
    """自定义评估示例"""
    print("\n=== 自定义评估示例 ===")
    
    # 创建自定义评估器
    evaluator = DeepResearchEvaluator(
        llm_provider="gemini"
    )
    
    # 模拟agent输出
    agent_output = {
        "related_works": """
        Recent advances in natural language processing have shown significant improvements in various tasks. 
        Transformer models (Vaswani et al., 2017) have revolutionized the field by introducing attention mechanisms. 
        BERT (Devlin et al., 2018) demonstrated the effectiveness of bidirectional training for language understanding. 
        GPT models (Radford et al., 2019) showed the power of autoregressive language modeling.
        """,
        "citations": [
            "Vaswani, A., et al. (2017). Attention is all you need. NIPS.",
            "Devlin, J., et al. (2018). BERT: Pre-training of deep bidirectional transformers. NAACL.",
            "Radford, A., et al. (2019). Language models are unsupervised multitask learners. OpenAI."
        ],
        "selected_papers": [
            {
                "text": ["Transformer architecture with attention mechanisms..."],
                "citation": "Vaswani, A., et al. (2017). Attention is all you need. NIPS."
            },
            {
                "text": ["BERT model for bidirectional language understanding..."],
                "citation": "Devlin, J., et al. (2018). BERT: Pre-training of deep bidirectional transformers. NAACL."
            }
        ]
    }
    
    # 执行评估
    evaluation_result = evaluator.evaluate_deep_research_agent(agent_output)
    
    # 打印详细结果
    print("引用可靠性指标:")
    print(f"  引用准确性: {evaluation_result.citation_reliability.citation_accuracy:.3f}")
    print(f"  引用相关性: {evaluation_result.citation_reliability.citation_relevance:.3f}")
    print(f"  引用时效性: {evaluation_result.citation_reliability.citation_timeliness:.3f}")
    print(f"  引用完整性: {evaluation_result.citation_reliability.citation_completeness:.3f}")
    print(f"  引用密度: {evaluation_result.citation_reliability.citation_density:.3f}")
    print(f"  引用覆盖度: {evaluation_result.citation_reliability.citation_coverage:.3f}")
    
    print("\n事实支持指标:")
    print(f"  事实准确性: {evaluation_result.factual_support.factual_accuracy:.3f}")
    print(f"  事实可验证性: {evaluation_result.factual_support.factual_verifiability:.3f}")
    print(f"  事实一致性: {evaluation_result.factual_support.factual_consistency:.3f}")
    print(f"  事实完整性: {evaluation_result.factual_support.factual_completeness:.3f}")
    print(f"  幻觉率: {evaluation_result.factual_support.hallucination_rate:.3f}")
    
    print(f"\n总体分数: {evaluation_result.overall_score:.3f}")
    
    return evaluation_result

def example_file_based_evaluation():
    """基于文件的评估示例"""
    print("\n=== 基于文件的评估示例 ===")
    
    # 创建示例输入文件
    input_data = [
        {
            "paper_id": "nlp_paper_1",
            "abstract": "This paper presents a new approach to sentiment analysis using deep learning techniques..."
        },
        {
            "paper_id": "cv_paper_1", 
            "abstract": "We propose a novel method for object detection in computer vision using convolutional neural networks..."
        },
        {
            "paper_id": "ml_paper_1",
            "abstract": "This study investigates the application of reinforcement learning in autonomous systems..."
        }
    ]
    
    input_file = "example_input.json"
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(input_data, f, ensure_ascii=False, indent=2)
    
    # 创建配置
    config = EvaluationConfig(
        llm_provider="gemini",
        breadth=5,
        depth=2,
        diversity=0.1
    )
    
    # 创建评估流水线
    pipeline = DeepResearchEvaluationPipeline(config)
    
    # 从文件执行评估
    results = pipeline.evaluate_from_file(input_file, "file_based_results.json")
    
    print(f"从文件处理了 {len(results)} 个摘要")
    
    # 清理临时文件
    os.remove(input_file)
    
    return results

def main():
    """主函数 - 运行所有示例"""
    print("Deep Research Agents评估系统使用示例")
    print("=" * 50)
    
    try:
        # 1. 单个摘要评估
        example_single_evaluation()
        
        # 2. 批量评估
        example_batch_evaluation()
        
        # 3. 基线对比
        example_baseline_comparison()
        
        # 4. 自定义评估
        example_custom_evaluation()
        
        # 5. 基于文件的评估
        example_file_based_evaluation()
        
        print("\n所有示例执行完成！")
        
    except Exception as e:
        print(f"执行示例时出错: {e}")
        print("请确保已正确配置API密钥和数据库连接")

if __name__ == "__main__":
    main()

