"""
Deep Research Agents评估器
基于MACG框架扩展，专门用于评估deep research agents在引用可靠性和事实支持方面的表现
"""

import json
import re
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from evaluation.agents.judge import Judge
from citegeist.generator import Generator
import json_repair
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

@dataclass
class CitationReliabilityMetrics:
    """引用可靠性指标"""
    citation_accuracy: float  # 引用准确性
    citation_relevance: float  # 引用相关性
    citation_timeliness: float  # 引用时效性
    citation_completeness: float  # 引用完整性
    citation_density: float  # 引用密度
    citation_coverage: float  # 引用覆盖度

@dataclass
class FactualSupportMetrics:
    """事实支持指标"""
    factual_accuracy: float  # 事实准确性
    factual_verifiability: float  # 事实可验证性
    factual_consistency: float  # 事实一致性
    factual_completeness: float  # 事实完整性
    hallucination_rate: float  # 幻觉率

@dataclass
class DeepResearchEvaluationResult:
    """Deep Research评估结果"""
    citation_reliability: CitationReliabilityMetrics
    factual_support: FactualSupportMetrics
    overall_score: float
    detailed_analysis: Dict[str, Any]

class DeepResearchEvaluator:
    """Deep Research Agents评估器"""
    
    def __init__(self, 
                 llm_provider: str = "gemini",
                 api_key: Optional[str] = None,
                 database_uri: Optional[str] = None,
                 database_token: Optional[str] = None):
        """
        初始化评估器
        
        Args:
            llm_provider: LLM提供商
            api_key: API密钥
            database_uri: 数据库URI
            database_token: 数据库令牌
        """
        self.judge_gemini = Judge(model="google/gemini-2.5-flash")
        self.judge_deepseek = Judge(model="deepseek-chat")
        
        # 初始化生成器用于对比
        self.generator = Generator(
            llm_provider=llm_provider,
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            model_name="google/gemini-2.5-flash",
            database_uri=database_uri or os.environ.get("MILVUS_URI", ""),
            database_token=database_token or os.environ.get("MILVUS_TOKEN", "")
        )
    
    def evaluate_citation_reliability(self, 
                                    related_work: str, 
                                    citations: List[str], 
                                    selected_papers: List[Dict]) -> CitationReliabilityMetrics:
        """
        评估引用可靠性
        
        Args:
            related_work: 相关工作文本
            citations: 引用列表
            selected_papers: 选中的论文列表
            
        Returns:
            CitationReliabilityMetrics: 引用可靠性指标
        """
        # 1. 引用准确性评估
        citation_accuracy = self._evaluate_citation_accuracy(related_work, citations, selected_papers)
        
        # 2. 引用相关性评估
        citation_relevance = self._evaluate_citation_relevance(related_work, citations, selected_papers)
        
        # 3. 引用时效性评估
        citation_timeliness = self._evaluate_citation_timeliness(citations, selected_papers)
        
        # 4. 引用完整性评估
        citation_completeness = self._evaluate_citation_completeness(citations, selected_papers)
        
        # 5. 引用密度评估
        citation_density = self._evaluate_citation_density(related_work, citations)
        
        # 6. 引用覆盖度评估
        citation_coverage = self._evaluate_citation_coverage(related_work, citations, selected_papers)
        
        return CitationReliabilityMetrics(
            citation_accuracy=citation_accuracy,
            citation_relevance=citation_relevance,
            citation_timeliness=citation_timeliness,
            citation_completeness=citation_completeness,
            citation_density=citation_density,
            citation_coverage=citation_coverage
        )
    
    def evaluate_factual_support(self, 
                               related_work: str, 
                               citations: List[str], 
                               selected_papers: List[Dict]) -> FactualSupportMetrics:
        """
        评估事实支持
        
        Args:
            related_work: 相关工作文本
            citations: 引用列表
            selected_papers: 选中的论文列表
            
        Returns:
            FactualSupportMetrics: 事实支持指标
        """
        # 1. 事实准确性评估
        factual_accuracy = self._evaluate_factual_accuracy(related_work, citations, selected_papers)
        
        # 2. 事实可验证性评估
        factual_verifiability = self._evaluate_factual_verifiability(related_work, citations, selected_papers)
        
        # 3. 事实一致性评估
        factual_consistency = self._evaluate_factual_consistency(related_work, citations, selected_papers)
        
        # 4. 事实完整性评估
        factual_completeness = self._evaluate_factual_completeness(related_work, citations, selected_papers)
        
        # 5. 幻觉率评估
        hallucination_rate = self._evaluate_hallucination_rate(related_work, citations, selected_papers)
        
        return FactualSupportMetrics(
            factual_accuracy=factual_accuracy,
            factual_verifiability=factual_verifiability,
            factual_consistency=factual_consistency,
            factual_completeness=factual_completeness,
            hallucination_rate=hallucination_rate
        )
    
    def _evaluate_citation_accuracy(self, related_work: str, citations: List[str], selected_papers: List[Dict]) -> float:
        """评估引用准确性"""
        # 提取引用句子
        cited_sentences = self._extract_cited_sentences(related_work)
        
        correct_citations = 0
        total_citations = 0
        
        for sentence in cited_sentences:
            # 检查引用格式是否正确
            citation_patterns = [
                r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)',  # (Author, 2024)
                r'[A-Z][a-z]+(?:\s+et\s+al\.)?\s*\(\d{4}\)',  # Author (2024)
            ]
            
            for pattern in citation_patterns:
                matches = re.findall(pattern, sentence)
                total_citations += len(matches)
                
                for match in matches:
                    # 验证引用是否在citations列表中
                    if any(match in citation for citation in citations):
                        correct_citations += 1
        
        return correct_citations / total_citations if total_citations > 0 else 0.0
    
    def _evaluate_citation_relevance(self, related_work: str, citations: List[str], selected_papers: List[Dict]) -> float:
        """评估引用相关性"""
        cited_sentences = self._extract_cited_sentences(related_work)
        relevant_citations = 0
        total_citations = 0
        
        for sentence in cited_sentences:
            # 提取句子中的引用
            citation_matches = re.findall(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)', sentence)
            
            for citation_match in citation_matches:
                total_citations += 1
                
                # 找到对应的论文
                paper_content = self._find_paper_content_by_citation(citation_match, citations, selected_papers)
                
                if paper_content:
                    # 使用LLM判断相关性
                    relevance_score = self._judge_citation_relevance(sentence, paper_content)
                    if relevance_score > 0.7:  # 阈值可调整
                        relevant_citations += 1
        
        return relevant_citations / total_citations if total_citations > 0 else 0.0
    
    def _evaluate_citation_timeliness(self, citations: List[str], selected_papers: List[Dict]) -> float:
        """评估引用时效性"""
        current_year = 2024  # 可配置
        recent_years = 3  # 最近3年
        
        recent_citations = 0
        total_citations = len(citations)
        
        for citation in citations:
            # 提取年份
            year_match = re.search(r'(\d{4})', citation)
            if year_match:
                year = int(year_match.group(1))
                if year >= current_year - recent_years:
                    recent_citations += 1
        
        return recent_citations / total_citations if total_citations > 0 else 0.0
    
    def _evaluate_citation_completeness(self, citations: List[str], selected_papers: List[Dict]) -> float:
        """评估引用完整性"""
        complete_citations = 0
        total_citations = len(citations)
        
        for citation in citations:
            # 检查引用是否包含必要信息：作者、年份、标题等
            has_author = bool(re.search(r'[A-Z][a-z]+', citation))
            has_year = bool(re.search(r'\d{4}', citation))
            has_title = len(citation) > 20  # 简单启发式
            
            if has_author and has_year and has_title:
                complete_citations += 1
        
        return complete_citations / total_citations if total_citations > 0 else 0.0
    
    def _evaluate_citation_density(self, related_work: str, citations: List[str]) -> float:
        """评估引用密度"""
        sentences = self._count_sentences(related_work)
        total_citations = len(citations)
        
        return total_citations / sentences if sentences > 0 else 0.0
    
    def _evaluate_citation_coverage(self, related_work: str, citations: List[str], selected_papers: List[Dict]) -> float:
        """评估引用覆盖度"""
        cited_sentences = self._extract_cited_sentences(related_work)
        total_sentences = len(self._count_sentences(related_work))
        
        return len(cited_sentences) / total_sentences if total_sentences > 0 else 0.0
    
    def _evaluate_factual_accuracy(self, related_work: str, citations: List[str], selected_papers: List[Dict]) -> float:
        """评估事实准确性"""
        cited_sentences = self._extract_cited_sentences(related_work)
        accurate_facts = 0
        total_facts = len(cited_sentences)
        
        for sentence in cited_sentences:
            # 找到对应的论文内容
            paper_content = self._find_paper_content_by_sentence(sentence, citations, selected_papers)
            
            if paper_content:
                # 使用双模型验证事实准确性
                accuracy_score = self._judge_factual_accuracy(sentence, paper_content)
                if accuracy_score > 0.7:  # 阈值可调整
                    accurate_facts += 1
        
        return accurate_facts / total_facts if total_facts > 0 else 0.0
    
    def _evaluate_factual_verifiability(self, related_work: str, citations: List[str], selected_papers: List[Dict]) -> float:
        """评估事实可验证性"""
        cited_sentences = self._extract_cited_sentences(related_work)
        verifiable_facts = 0
        total_facts = len(cited_sentences)
        
        for sentence in cited_sentences:
            # 检查是否有对应的引用
            has_citation = bool(re.search(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)', sentence))
            
            if has_citation:
                # 检查引用是否可验证
                paper_content = self._find_paper_content_by_sentence(sentence, citations, selected_papers)
                if paper_content and len(paper_content) > 100:  # 有足够的内容进行验证
                    verifiable_facts += 1
        
        return verifiable_facts / total_facts if total_facts > 0 else 0.0
    
    def _evaluate_factual_consistency(self, related_work: str, citations: List[str], selected_papers: List[Dict]) -> float:
        """评估事实一致性"""
        cited_sentences = self._extract_cited_sentences(related_work)
        
        if len(cited_sentences) < 2:
            return 1.0  # 只有一个句子，认为一致
        
        consistent_pairs = 0
        total_pairs = 0
        
        for i in range(len(cited_sentences)):
            for j in range(i + 1, len(cited_sentences)):
                total_pairs += 1
                
                # 检查两个句子是否一致
                consistency_score = self._judge_factual_consistency(cited_sentences[i], cited_sentences[j])
                if consistency_score > 0.7:  # 阈值可调整
                    consistent_pairs += 1
        
        return consistent_pairs / total_pairs if total_pairs > 0 else 1.0
    
    def _evaluate_factual_completeness(self, related_work: str, citations: List[str], selected_papers: List[Dict]) -> float:
        """评估事实完整性"""
        # 这里可以实现更复杂的逻辑，比如检查是否遗漏了重要信息
        # 简化版本：检查是否有足够的引用支持主要观点
        main_topics = self._extract_main_topics(related_work)
        supported_topics = 0
        
        for topic in main_topics:
            # 检查是否有引用支持这个主题
            if self._has_supporting_citations(topic, related_work, citations):
                supported_topics += 1
        
        return supported_topics / len(main_topics) if main_topics else 1.0
    
    def _evaluate_hallucination_rate(self, related_work: str, citations: List[str], selected_papers: List[Dict]) -> float:
        """评估幻觉率"""
        cited_sentences = self._extract_cited_sentences(related_work)
        hallucinated_sentences = 0
        total_sentences = len(cited_sentences)
        
        for sentence in cited_sentences:
            # 检查句子是否包含无法验证的信息
            if self._is_hallucinated(sentence, citations, selected_papers):
                hallucinated_sentences += 1
        
        return hallucinated_sentences / total_sentences if total_sentences > 0 else 0.0
    
    # 辅助方法
    def _extract_cited_sentences(self, text: str) -> List[str]:
        """提取包含引用的句子"""
        sentences = re.split(r'[.!?]+', text)
        cited_sentences = []
        
        for sentence in sentences:
            if re.search(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)', sentence):
                cited_sentences.append(sentence.strip())
        
        return cited_sentences
    
    def _count_sentences(self, text: str) -> List[str]:
        """计算句子数量"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_paper_content_by_citation(self, citation: str, citations: List[str], selected_papers: List[Dict]) -> Optional[str]:
        """根据引用找到对应的论文内容"""
        for i, cit in enumerate(citations):
            if citation in cit and i < len(selected_papers):
                paper = selected_papers[i]
                if 'text' in paper:
                    return '\n'.join(paper['text']) if isinstance(paper['text'], list) else str(paper['text'])
        return None
    
    def _find_paper_content_by_sentence(self, sentence: str, citations: List[str], selected_papers: List[Dict]) -> Optional[str]:
        """根据句子找到对应的论文内容"""
        # 提取句子中的引用
        citation_matches = re.findall(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)', sentence)
        
        for citation_match in citation_matches:
            content = self._find_paper_content_by_citation(citation_match, citations, selected_papers)
            if content:
                return content
        
        return None
    
    def _judge_citation_relevance(self, sentence: str, paper_content: str) -> float:
        """判断引用相关性"""
        try:
            response = self.judge_gemini.get_pair_score_new(paper_content, sentence)
            return 1.0 if response.lower() == "yes" else 0.0
        except Exception as e:
            logger.error(f"Error judging citation relevance: {e}")
            return 0.0
    
    def _judge_factual_accuracy(self, sentence: str, paper_content: str) -> float:
        """判断事实准确性"""
        try:
            response_gemini = self.judge_gemini.get_pair_score_new(paper_content, sentence)
            response_deepseek = self.judge_deepseek.get_pair_score_new(paper_content, sentence)
            
            # 两个模型都认为准确才算准确
            if response_gemini.lower() == "yes" and response_deepseek.lower() == "yes":
                return 1.0
            elif response_gemini.lower() == "yes" or response_deepseek.lower() == "yes":
                return 0.5
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Error judging factual accuracy: {e}")
            return 0.0
    
    def _judge_factual_consistency(self, sentence1: str, sentence2: str) -> float:
        """判断事实一致性"""
        # 简化实现，实际可以使用更复杂的逻辑
        try:
            prompt = f"判断以下两个句子是否在事实上一致：\n句子1: {sentence1}\n句子2: {sentence2}\n请回答'是'或'否'。"
            response = self.judge_gemini.llm_client.get_completion(prompt)
            return 1.0 if "是" in response else 0.0
        except Exception as e:
            logger.error(f"Error judging factual consistency: {e}")
            return 0.0
    
    def _extract_main_topics(self, text: str) -> List[str]:
        """提取主要主题"""
        # 简化实现，实际可以使用更复杂的NLP技术
        sentences = self._count_sentences(text)
        return sentences[:3]  # 取前3个句子作为主要主题
    
    def _has_supporting_citations(self, topic: str, related_work: str, citations: List[str]) -> bool:
        """检查主题是否有支持引用"""
        # 简化实现
        return bool(re.search(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)', topic))
    
    def _is_hallucinated(self, sentence: str, citations: List[str], selected_papers: List[Dict]) -> bool:
        """判断句子是否为幻觉"""
        # 检查是否有引用支持
        has_citation = bool(re.search(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)', sentence))
        
        if not has_citation:
            return True
        
        # 检查引用是否可验证
        paper_content = self._find_paper_content_by_sentence(sentence, citations, selected_papers)
        if not paper_content:
            return True
        
        # 使用LLM判断是否为幻觉
        try:
            response = self.judge_gemini.get_pair_score_new(paper_content, sentence)
            return response.lower() != "yes"
        except Exception as e:
            logger.error(f"Error judging hallucination: {e}")
            return True
    
    def evaluate_deep_research_agent(self, 
                                   agent_output: Dict[str, Any], 
                                   ground_truth: Optional[Dict[str, Any]] = None) -> DeepResearchEvaluationResult:
        """
        评估deep research agent的输出
        
        Args:
            agent_output: agent的输出，包含related_work, citations, selected_papers等
            ground_truth: 真实标签（可选）
            
        Returns:
            DeepResearchEvaluationResult: 评估结果
        """
        related_work = agent_output.get("related_works", "")
        citations = agent_output.get("citations", [])
        selected_papers = agent_output.get("selected_papers", [])
        
        # 评估引用可靠性
        citation_reliability = self.evaluate_citation_reliability(related_work, citations, selected_papers)
        
        # 评估事实支持
        factual_support = self.evaluate_factual_support(related_work, citations, selected_papers)
        
        # 计算总体分数
        overall_score = self._calculate_overall_score(citation_reliability, factual_support)
        
        # 详细分析
        detailed_analysis = {
            "citation_analysis": {
                "total_citations": len(citations),
                "citation_patterns": self._analyze_citation_patterns(related_work),
                "citation_errors": self._identify_citation_errors(related_work, citations)
            },
            "factual_analysis": {
                "total_claims": len(self._extract_cited_sentences(related_work)),
                "verification_coverage": self._calculate_verification_coverage(related_work, citations, selected_papers),
                "consistency_issues": self._identify_consistency_issues(related_work, citations, selected_papers)
            }
        }
        
        return DeepResearchEvaluationResult(
            citation_reliability=citation_reliability,
            factual_support=factual_support,
            overall_score=overall_score,
            detailed_analysis=detailed_analysis
        )
    
    def _calculate_overall_score(self, citation_reliability: CitationReliabilityMetrics, 
                               factual_support: FactualSupportMetrics) -> float:
        """计算总体分数"""
        # 权重可调整
        citation_weight = 0.4
        factual_weight = 0.6
        
        citation_score = (
            citation_reliability.citation_accuracy * 0.3 +
            citation_reliability.citation_relevance * 0.3 +
            citation_reliability.citation_timeliness * 0.2 +
            citation_reliability.citation_completeness * 0.2
        )
        
        factual_score = (
            factual_support.factual_accuracy * 0.3 +
            factual_support.factual_verifiability * 0.3 +
            factual_support.factual_consistency * 0.2 +
            (1 - factual_support.hallucination_rate) * 0.2
        )
        
        return citation_score * citation_weight + factual_score * factual_weight
    
    def _analyze_citation_patterns(self, text: str) -> Dict[str, Any]:
        """分析引用模式"""
        patterns = {
            "author_year": len(re.findall(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)', text)),
            "author_year_inline": len(re.findall(r'[A-Z][a-z]+(?:\s+et\s+al\.)?\s*\(\d{4}\)', text)),
            "et_al_usage": len(re.findall(r'et\s+al\.', text))
        }
        return patterns
    
    def _identify_citation_errors(self, text: str, citations: List[str]) -> List[str]:
        """识别引用错误"""
        errors = []
        
        # 检查格式错误
        malformed_citations = re.findall(r'\([^)]*\d{4}[^)]*\)', text)
        for citation in malformed_citations:
            if not re.match(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)', citation):
                errors.append(f"格式错误: {citation}")
        
        # 检查引用是否在列表中
        cited_refs = re.findall(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)', text)
        for ref in cited_refs:
            if not any(ref in citation for citation in citations):
                errors.append(f"未找到引用: {ref}")
        
        return errors
    
    def _calculate_verification_coverage(self, text: str, citations: List[str], selected_papers: List[Dict]) -> float:
        """计算验证覆盖率"""
        cited_sentences = self._extract_cited_sentences(text)
        verifiable_sentences = 0
        
        for sentence in cited_sentences:
            paper_content = self._find_paper_content_by_sentence(sentence, citations, selected_papers)
            if paper_content and len(paper_content) > 100:
                verifiable_sentences += 1
        
        return verifiable_sentences / len(cited_sentences) if cited_sentences else 0.0
    
    def _identify_consistency_issues(self, text: str, citations: List[str], selected_papers: List[Dict]) -> List[str]:
        """识别一致性问题"""
        issues = []
        cited_sentences = self._extract_cited_sentences(text)
        
        for i in range(len(cited_sentences)):
            for j in range(i + 1, len(cited_sentences)):
                consistency_score = self._judge_factual_consistency(cited_sentences[i], cited_sentences[j])
                if consistency_score < 0.3:  # 阈值可调整
                    issues.append(f"一致性问题: 句子{i+1}和句子{j+1}可能存在矛盾")
        
        return issues
