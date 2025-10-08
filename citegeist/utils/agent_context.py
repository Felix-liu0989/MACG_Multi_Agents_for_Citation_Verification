"""
Agent通信上下文管理模块
用于在多个agent之间传递完整的上下文信息,避免信息损失
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
from pathlib import Path


@dataclass
class PaperContext:
    """单篇论文的完整上下文"""
    paper_id: int
    arxiv_id: str
    title: str
    abstract: str
    summary: str
    citation: str
    full_text_segments: List[str] = field(default_factory=list)  # 完整的页面内容
    cite_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "summary": self.summary,
            "citation": self.citation,
            "full_text_segments": self.full_text_segments,
            "cite_ids": self.cite_ids,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PaperContext':
        return cls(**data)


@dataclass
class DAGContext:
    """DAG结构的完整上下文"""
    dimensions: List[str]
    grouped_papers: Dict[str, List[Dict]]  # dimension -> papers
    dag_structure: Dict[str, Any] = field(default_factory=dict)  # DAG的层次结构
    visualization_path: Optional[str] = None
    topic: str = ""
    outline: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "dimensions": self.dimensions,
            "grouped_papers": self.grouped_papers,
            "dag_structure": self.dag_structure,
            "visualization_path": self.visualization_path,
            "topic": self.topic,
            "outline": self.outline
        }


@dataclass
class CitationContext:
    """引用验证的完整上下文"""
    citations: List[Dict[str, Any]]  # citation_text, paper_id
    cited_sentences: List[str] = field(default_factory=list)
    quotes_with_citation_info: Dict[str, List[str]] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    error_types: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "citations": self.citations,
            "cited_sentences": self.cited_sentences,
            "quotes_with_citation_info": self.quotes_with_citation_info,
            "validation_results": self.validation_results,
            "error_types": self.error_types
        }


@dataclass
class AgentCommunicationContext:
    """
    多Agent通信的共享上下文
    这个对象在各个agent之间传递,确保信息不丢失
    """
    # 基础信息
    source_paper_abstract: str
    arxiv_id: str
    
    # 检索到的论文完整上下文
    papers: List[PaperContext] = field(default_factory=list)
    
    # DAG构建上下文
    dag_context: Optional[DAGContext] = None
    
    # Related Work生成上下文
    related_work_original: str = ""
    related_work_revised: str = ""
    
    # Feedback上下文
    feedback: str = ""
    feedback_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Citation上下文
    citation_context: Optional[CitationContext] = None
    
    # 参数配置
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Agent执行历史
    agent_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_agent_execution(self, agent_name: str, input_data: Dict, output_data: Dict, 
                           execution_time: float = 0.0, metadata: Dict = None):
        """记录每个agent的执行过程"""
        self.agent_history.append({
            "agent_name": agent_name,
            "timestamp": execution_time,
            "input_summary": self._summarize_data(input_data),
            "output_summary": self._summarize_data(output_data),
            "metadata": metadata or {}
        })
    
    def _summarize_data(self, data: Dict) -> Dict:
        """简化数据用于记录(避免存储大量文本)"""
        summary = {}
        for key, value in data.items():
            if isinstance(value, str):
                summary[key] = f"{value[:100]}..." if len(value) > 100 else value
            elif isinstance(value, list):
                summary[key] = f"List[{len(value)} items]"
            elif isinstance(value, dict):
                summary[key] = f"Dict[{len(value)} keys]"
            else:
                summary[key] = str(value)
        return summary
    
    def get_paper_by_id(self, paper_id: int) -> Optional[PaperContext]:
        """根据paper_id获取完整的论文上下文"""
        for paper in self.papers:
            if paper.paper_id == paper_id:
                return paper
        return None
    
    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[PaperContext]:
        """根据arxiv_id获取完整的论文上下文"""
        for paper in self.papers:
            if paper.arxiv_id == arxiv_id:
                return paper
        return None
    
    def save_to_file(self, filepath: Path):
        """保存完整上下文到文件"""
        data = {
            "source_paper_abstract": self.source_paper_abstract,
            "arxiv_id": self.arxiv_id,
            "papers": [p.to_dict() for p in self.papers],
            "dag_context": self.dag_context.to_dict() if self.dag_context else None,
            "related_work_original": self.related_work_original,
            "related_work_revised": self.related_work_revised,
            "feedback": self.feedback,
            "feedback_metadata": self.feedback_metadata,
            "citation_context": self.citation_context.to_dict() if self.citation_context else None,
            "config": self.config,
            "agent_history": self.agent_history
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'AgentCommunicationContext':
        """从文件加载完整上下文"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        ctx = cls(
            source_paper_abstract=data["source_paper_abstract"],
            arxiv_id=data["arxiv_id"]
        )
        ctx.papers = [PaperContext.from_dict(p) for p in data.get("papers", [])]
        
        if data.get("dag_context"):
            ctx.dag_context = DAGContext(**data["dag_context"])
        
        ctx.related_work_original = data.get("related_work_original", "")
        ctx.related_work_revised = data.get("related_work_revised", "")
        ctx.feedback = data.get("feedback", "")
        ctx.feedback_metadata = data.get("feedback_metadata", {})
        
        if data.get("citation_context"):
            ctx.citation_context = CitationContext(**data["citation_context"])
        
        ctx.config = data.get("config", {})
        ctx.agent_history = data.get("agent_history", [])
        
        return ctx
    
    def get_context_for_agent(self, agent_name: str) -> Dict[str, Any]:
        """
        为特定agent提供所需的上下文
        这样可以确保每个agent都能访问到需要的所有信息
        """
        base_context = {
            "source_abstract": self.source_paper_abstract,
            "arxiv_id": self.arxiv_id,
            "config": self.config
        }
        
        if agent_name == "dag_builder":
            return {
                **base_context,
                "papers": [p.to_dict() for p in self.papers]
            }
        
        elif agent_name == "related_work_generator":
            return {
                **base_context,
                "papers": [p.to_dict() for p in self.papers],
                "dag_context": self.dag_context.to_dict() if self.dag_context else None
            }
        
        elif agent_name == "feedback_agent":
            return {
                **base_context,
                "related_work": self.related_work_original,
                "papers": [p.to_dict() for p in self.papers],
                "dag_context": self.dag_context.to_dict() if self.dag_context else None,
                "citations": self.citation_context.citations if self.citation_context else []
            }
        
        elif agent_name == "revision_agent":
            return {
                **base_context,
                "related_work_original": self.related_work_original,
                "feedback": self.feedback,
                "feedback_metadata": self.feedback_metadata,
                "papers": [p.to_dict() for p in self.papers],  # 包含完整的page content
                "dag_context": self.dag_context.to_dict() if self.dag_context else None,
                "citations": self.citation_context.citations if self.citation_context else []
            }
        
        elif agent_name == "validation_agent":
            return {
                **base_context,
                "related_work": self.related_work_revised or self.related_work_original,
                "papers": [p.to_dict() for p in self.papers],  # 完整的论文内容用于验证
                "citations": self.citation_context.citations if self.citation_context else []
            }
        
        return base_context
