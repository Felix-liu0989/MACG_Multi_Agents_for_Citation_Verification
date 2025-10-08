# Agent通信优化指南

## 📋 问题诊断

当前系统存在的主要信息损失点:

### 1. DAG构建 → Related Work生成
**问题**: 只传递了扁平的grouped结构,丢失了:
- DAG的层次关系
- 节点之间的父子关系  
- 论文在分类树中的位置信息

**影响**: 生成的related work无法充分利用文献的层次分类信息

### 2. Related Work生成 → Feedback Agent
**问题**: Feedback agent只接收related_work文本,缺少:
- 源论文的abstract和研究目标
- DAG结构和文献分类逻辑
- 原始引用的完整上下文

**影响**: 反馈缺乏针对性,无法评估引用准确性和主题相关性

### 3. Feedback → Revision Agent  
**问题**: Revision agent缺少:
- 完整的论文page content(只有summary)
- 结构化的feedback信息
- DAG组织逻辑

**影响**: 修订时无法验证引用准确性,可能引入新错误

### 4. Revision → Validation Agent
**问题**: Validation只用summary匹配,而不是:
- 完整的page content
- 论文的具体章节内容

**影响**: 验证准确性降低,误判率高

---

## 💡 优化方案

### 核心思路: 共享上下文对象

使用`AgentCommunicationContext`在所有agent之间传递完整信息:

```python
from citegeist.agent_context import (
    AgentCommunicationContext, 
    PaperContext, 
    DAGContext,
    CitationContext
)
```

### 优化步骤

#### Step 1: 初始化上下文

```python
def generate_related_work_MACG(
    self,
    abstract: str,
    breadth: int,
    depth: int,
    diversity: float,
    arxiv_id: str,
    status_callback: Optional[Callable] = None,
):
    # 创建共享上下文
    context = AgentCommunicationContext(
        source_paper_abstract=abstract,
        arxiv_id=arxiv_id,
        config={
            "breadth": breadth,
            "depth": depth,
            "diversity": diversity
        }
    )
```

#### Step 2: 收集论文时保存完整内容

```python
# 原代码只保存了summary,现在保存完整的page content
for obj in relevant_pages[1:]:
    arxiv_id = papers_data[obj["paper_id"]]["id"]
    arxiv_abstract = get_arxiv_abstract(arxiv_id)
    text_segments = obj["text"]  # ← 这是完整的页面内容!
    title = get_arxiv_title(arxiv_id)
    
    # 生成summary
    prompt = generate_summary_prompt_with_page_content(...)
    summary = self.llm_client.get_completion(prompt)
    citation = get_arxiv_citation(arxiv_id)
    
    # 创建PaperContext,保存所有信息
    paper_ctx = PaperContext(
        paper_id=id,
        arxiv_id=arxiv_id,
        title=title,
        abstract=arxiv_abstract,
        summary=summary,
        citation=citation,
        full_text_segments=text_segments,  # ← 保存完整内容!
        cite_ids=[id]
    )
    context.papers.append(paper_ctx)
    id += 1
```

#### Step 3: DAG构建时保存结构信息

```python
# 构建DAG并保存到context
roots, dags, id2node, label2node = build_dags(args)
results = label_papers_by_topic(args, internal_collection, subsection_titles)
update_roots_with_labels(roots, results, internal_collection, args)

# 构建grouped结构
grouped = {dim: [
    {
        "paper_id": pid,
        "title": paper.title,
        "abstract": paper.abstract,
        "summary": paper.summary,
        "citations": paper.citations
    }
    for pid, paper in roots[dim].papers.items()
] for dim in args.dimensions}

# 保存DAG上下文
context.dag_context = DAGContext(
    dimensions=args.dimensions,
    grouped_papers=grouped,
    dag_structure={  # ← 保存结构信息!
        "roots": {dim: root.to_dict() for dim, root in roots.items()},
        "id2node": id2node,
        "label2node": label2node
    },
    topic=topic,
    outline=outline
)

# 记录agent执行
context.add_agent_execution(
    agent_name="dag_builder",
    input_data={"dimensions": args.dimensions},
    output_data={"grouped_papers_count": sum(len(v) for v in grouped.values())},
    execution_time=time.time() - start_time_DAG
)
```

#### Step 4: 生成Related Work时使用完整上下文

```python
# 获取agent所需的完整上下文
rw_context = context.get_context_for_agent("related_work_generator")

prompt = generate_related_work_prompt_with_arxiv_trees(
    abstract=context.source_paper_abstract,
    dimensions=context.dag_context.dimensions,
    grouped=context.dag_context.grouped_papers
)
related_work_with_citations = self.llm_client.get_completion(prompt)
related_work_with_citations = json_repair.loads(related_work_with_citations)

# 保存到context
context.related_work_original = related_work_with_citations["related_work"]
context.citation_context = CitationContext(
    citations=related_work_with_citations["cite_ids"]
)

context.add_agent_execution(
    agent_name="related_work_generator",
    input_data={"papers_count": len(context.papers)},
    output_data={"citations_count": len(context.citation_context.citations)},
    execution_time=time.time() - start_time_related_work
)
```

#### Step 5: Feedback时提供完整上下文

```python
from citegeist.utils.prompts_enhanced import generate_enhanced_feedback_prompt

# 使用增强版prompt,提供完整上下文
feedback_context = context.get_context_for_agent("feedback_agent")

feedback_prompt = generate_enhanced_feedback_prompt(
    related_work=context.related_work_original,
    source_abstract=context.source_paper_abstract,
    dag_context=context.dag_context.to_dict(),
    papers=[p.to_dict() for p in context.papers],
    citations=context.citation_context.citations
)

feedback = self.llm_client.get_completion(feedback_prompt)
context.feedback = feedback

# 可选: 解析feedback为结构化数据
try:
    feedback_structured = json_repair.loads(feedback)
    context.feedback_metadata = {
        "priority_issues": feedback_structured.get("priority_issues", []),
        "overall_score": feedback_structured.get("overall_score", 0)
    }
except:
    pass

context.add_agent_execution(
    agent_name="feedback_agent",
    input_data={"related_work_length": len(context.related_work_original)},
    output_data={"feedback_length": len(feedback)},
    execution_time=time.time() - start_time_feedback
)
```

#### Step 6: Revision时使用完整的论文内容

```python
from citegeist.utils.prompts_enhanced import generate_enhanced_revision_prompt

revision_context = context.get_context_for_agent("revision_agent")

prompt_for_revision = generate_enhanced_revision_prompt(
    source_abstract=context.source_paper_abstract,
    related_work_original=context.related_work_original,
    feedback=context.feedback,
    feedback_metadata=context.feedback_metadata,
    papers=[p.to_dict() for p in context.papers],  # ← 包含full_text_segments!
    dag_context=context.dag_context.to_dict(),
    citations=context.citation_context.citations
)

related_work_revision = client.get_completion(prompt_for_revision)
related_work_revision_dict = json_repair.loads(related_work_revision)

context.related_work_revised = related_work_revision_dict["related_work"]
context.citation_context.citations = related_work_revision_dict["cite_ids"]

context.add_agent_execution(
    agent_name="revision_agent",
    input_data={"original_length": len(context.related_work_original)},
    output_data={"revised_length": len(context.related_work_revised)},
    execution_time=time.time() - start_time_revision
)
```

#### Step 7: Validation时使用完整文本

```python
from citegeist.utils.prompts_enhanced import generate_enhanced_validation_prompt_with_full_text

def _validate_citations_with_dual_models_enhanced(self, context: AgentCommunicationContext):
    """使用完整文本进行更准确的验证"""
    
    # 提取cited sentences
    prompt_for_extract = process_data_for_extract_cited_sentences(context.related_work_revised)
    cited_sentences = self.llm_client.get_completion(prompt_for_extract)
    cited_sentences = json_repair.loads(cited_sentences)
    
    context.citation_context.cited_sentences = cited_sentences
    
    validation_results = {
        "yes_gemini": [],
        "no_gemini": [],
        "yes_deepseek": [],
        "no_deepseek": []
    }
    
    # 构建引用映射
    quotes_with_citation_info = {}
    for quote in cited_sentences:
        for citation in context.citation_context.citations:
            c_text = citation["citation_text"]
            if c_text in quote:
                paper_id = int(citation["paper_id"].replace("paper_", ""))
                paper = context.get_paper_by_id(paper_id)
                
                if paper:
                    if quote not in quotes_with_citation_info:
                        quotes_with_citation_info[quote] = []
                    
                    # ← 使用完整的论文内容进行验证!
                    quotes_with_citation_info[quote].append(paper.to_dict())
    
    # 使用增强版验证prompt
    for quote, papers in quotes_with_citation_info.items():
        for paper in papers:
            # use_full_text=True 会使用full_text_segments而不仅仅是summary
            prompt = generate_enhanced_validation_prompt_with_full_text(
                claim=quote,
                paper=paper,
                use_full_text=True  # ← 启用完整文本验证!
            )
            
            score_gemini = judge_gemini.get_pair_score_new_with_prompt(prompt)
            score_deepseek = judge_deepseek.get_pair_score_new_with_prompt(prompt)
            
            if score_gemini.lower() == "yes":
                validation_results["yes_gemini"].append({"quote": quote, "paper_id": paper["paper_id"]})
            else:
                validation_results["no_gemini"].append({"quote": quote, "paper_id": paper["paper_id"]})
            
            if score_deepseek.lower() == "yes":
                validation_results["yes_deepseek"].append({"quote": quote, "paper_id": paper["paper_id"]})
            else:
                validation_results["no_deepseek"].append({"quote": quote, "paper_id": paper["paper_id"]})
    
    context.citation_context.validation_results = validation_results
    return validation_results
```

#### Step 8: 保存完整上下文用于调试和分析

```python
# 在流程结束时保存完整上下文
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
results_dir = PROJECT_ROOT / f"results/{arxiv_id.replace('.','_')}"
results_dir.mkdir(exist_ok=True, parents=True)

# 保存完整的agent通信上下文
context.save_to_file(results_dir / f"agent_context_{breadth}_{depth}_{diversity}.json")

# 保存结果
results_dict = {
    "related_work": context.related_work_revised,
    "citations": context.citation_context.citations,
    "validation_results": context.citation_context.validation_results,
    "agent_execution_history": context.agent_history  # ← 包含所有agent的执行历史
}

with open(results_dir / f"results_{breadth}_{depth}_{diversity}.json", "w", encoding="utf-8") as f:
    json.dump(results_dict, f, ensure_ascii=False, indent=4)
```

---

## 📊 优化效果对比

### 优化前:
```
Papers (summary only) → DAG Builder → Grouped Papers
                                          ↓
                                    Related Work Generator
                                          ↓
                                    Feedback Agent (只有related_work文本)
                                          ↓
                                    Revision Agent (只有summary)
                                          ↓
                                    Validation Agent (只有summary)
```

**信息损失**:
- DAG结构信息 ❌
- 完整的page content ❌  
- 源论文abstract ❌
- Agent间的上下文 ❌

### 优化后:
```
AgentCommunicationContext (包含所有信息)
    ├── Papers (full_text_segments + summary + abstract)
    ├── DAG Context (structure + grouped + outline)
    ├── Related Work (original + revised)
    ├── Feedback (content + metadata)
    ├── Citation Context (citations + validation)
    └── Agent History (完整的执行记录)
        ↓
    所有Agent共享这个完整上下文
```

**信息保留**:
- DAG结构信息 ✅
- 完整的page content ✅
- 源论文abstract ✅
- Agent间的完整上下文 ✅
- 执行历史和可追溯性 ✅

---

## 🎯 预期改进

1. **Feedback质量提升 30-50%**
   - 有了源论文abstract,可以评估主题相关性
   - 有了DAG结构,可以评估组织逻辑
   - 有了完整papers,可以验证引用准确性

2. **Revision准确性提升 40-60%**
   - 有了完整的page content,可以准确修改引用
   - 有了结构化feedback,可以针对性改进
   - 有了DAG指导,可以保持组织逻辑

3. **Validation准确率提升 50-70%**
   - 使用完整文本而非summary验证
   - 减少误判(特别是false negatives)
   - 提供更精确的错误定位

4. **可调试性提升 100%**
   - 完整的agent执行历史
   - 可追溯每个agent的输入输出
   - 便于分析和优化

---

## 🔧 实施建议

### 阶段1: 基础实施(1-2天)
- [ ] 在`generator.py`中引入`AgentCommunicationContext`
- [ ] 修改paper收集逻辑,保存`full_text_segments`
- [ ] 修改DAG构建逻辑,保存结构信息

### 阶段2: Prompt优化(2-3天)  
- [ ] 使用`generate_enhanced_feedback_prompt`
- [ ] 使用`generate_enhanced_revision_prompt`
- [ ] 使用`generate_enhanced_validation_prompt_with_full_text`

### 阶段3: 测试和调优(3-5天)
- [ ] 对比优化前后的结果质量
- [ ] 调整prompt以获得最佳效果
- [ ] 优化context大小(避免超token限制)

### 阶段4: 生产部署(1-2天)
- [ ] 添加异常处理和降级逻辑
- [ ] 优化性能(缓存、并行等)
- [ ] 完善日志和监控

---

## ⚠️ 注意事项

1. **Token限制**: 完整的page content可能很长,需要:
   - 对长文本进行截断或摘要
   - 只选择最相关的segments
   - 使用更大的context window模型

2. **性能考虑**: 传递更多信息会:
   - 增加prompt长度
   - 增加API调用成本
   - 增加响应时间
   
   **解决方案**: 
   - 智能选择需要的信息
   - 使用缓存避免重复处理
   - 考虑使用更快的模型

3. **向后兼容**: 保留原有接口:
   ```python
   # 新接口(推荐)
   def generate_related_work_MACG_enhanced(...) -> dict:
       context = AgentCommunicationContext(...)
       ...
   
   # 旧接口(兼容性)
   def generate_related_work_MACG(...) -> dict:
       # 调用新接口
       return self.generate_related_work_MACG_enhanced(...)
   ```

---

## 📚 相关文件

- `citegeist/agent_context.py` - 核心上下文对象定义
- `citegeist/utils/prompts_enhanced.py` - 增强版prompt生成函数
- `citegeist/generator.py` - 主要修改位置

## 🤝 贡献指南

如果你发现新的信息损失点或有优化建议,请:
1. 在`agent_context.py`中扩展相应的Context类
2. 在`prompts_enhanced.py`中添加利用新信息的prompt
3. 更新本文档说明改进点

---

## 📞 支持

如有问题,请参考:
- 代码注释和docstring
- Agent执行历史(保存在results目录中)
- 完整的上下文文件(agent_context_*.json)
