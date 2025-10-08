"""
增强版的prompt生成函数
利用AgentCommunicationContext提供更丰富的上下文信息,减少信息损失
"""

from typing import Dict, List, Any


def generate_enhanced_feedback_prompt(
    related_work: str,
    source_abstract: str,
    dag_context: Dict[str, Any],
    papers: List[Dict[str, Any]],
    citations: List[Dict[str, Any]]
) -> str:
    """
    增强版的feedback prompt,包含更多上下文信息
    
    相比原版,增加了:
    1. 源论文的abstract(了解研究主题和目标)
    2. DAG结构信息(了解文献分类逻辑)
    3. 完整的papers信息(验证引用准确性)
    4. citations信息(检查引用是否合理)
    """
    
    # 构建DAG结构概述
    dag_summary = ""
    if dag_context and "dimensions" in dag_context:
        dag_summary = f"\n**文献分类结构:**\n"
        dag_summary += f"本篇Related Work按照以下{len(dag_context['dimensions'])}个维度组织:\n"
        for dim in dag_context['dimensions']:
            papers_in_dim = dag_context.get('grouped_papers', {}).get(dim, [])
            dag_summary += f"- {dim}: {len(papers_in_dim)}篇相关论文\n"
    
    # 构建论文库概述
    papers_summary = f"\n**可用的参考文献({len(papers)}篇):**\n"
    for i, paper in enumerate(papers[:], 1):  # 只显示前10篇避免prompt过长
        papers_summary += f"{i}. {paper.get('title', 'N/A')} (paper_id: {paper.get('paper_id', 'N/A')})\n"
    if len(papers) > 10:
        papers_summary += f"... 以及其他{len(papers)-10}篇论文\n"
    
    # 构建引用使用情况
    citation_summary = f"\n**当前引用情况:**\n"
    citation_summary += f"- 共使用了{len(citations)}处引用\n"
    cited_paper_ids = set([c.get('paper_id', '').replace('paper_', '') for c in citations if c.get('paper_id')])
    citation_summary += f"- 引用了{len(cited_paper_ids)}篇不同的论文\n"
    
    prompt = f"""
你是一位经验丰富的学术写作专家。请对以下Related Work部分提供详细的反馈意见。

**源论文摘要:**
{source_abstract}

{dag_summary}

{papers_summary}

{citation_summary}

**Related Work内容:**
{related_work}

请从以下几个方面提供反馈:

1. **结构逻辑性评估:**
   - 是否遵循了"问题→方法分类→局限性→创新点"的逻辑?
   - 各小节之间的过渡是否自然?
   - 是否充分利用了上述文献分类结构?

2. **批判性分析评估:**
   - 是否对引用的方法进行了批判性分析?
   - 是否指出了现有方法的局限性?
   - 是否将这些局限性与源论文的创新点联系起来?

3. **引用准确性评估:**
   - 引用的论文是否在可用的参考文献列表中?
   - 引用的数量和分布是否合理?
   - 是否存在过度引用或引用不足的情况?
   - 引用是否准确反映了被引论文的贡献?

4. **语言简洁性评估:**
   - 是否存在冗余描述?
   - 是否可以合并相似的引用?
   - 核心观点是否突出?

5. **创新点对比评估:**
   - 是否清晰地说明了源论文相比现有工作的优势?
   - 是否在结论部分直接说明了为什么源论文的方法更好?

请提供具体的改进建议,包括:
- 需要调整的段落或句子
- 建议的修改方向
- 需要增加或删除的内容

**输出格式:**
请以结构化的方式输出反馈,包括:
1. 总体评价(优点和不足)
2. 具体问题列表(每个问题包括:位置、问题描述、改进建议)
3. 优先级排序(哪些问题最需要优先解决)
"""
    return prompt


def generate_enhanced_revision_prompt(
    source_abstract: str,
    related_work_original: str,
    feedback: str,
    feedback_metadata: Dict[str, Any],
    papers: List[Dict[str, Any]],
    dag_context: Dict[str, Any],
    citations: List[Dict[str, Any]]
) -> str:
    """
    增强版的revision prompt,提供完整的上下文信息
    
    相比原版,增加了:
    1. feedback_metadata(反馈的结构化信息)
    2. 完整的papers信息(包括full_text_segments,用于验证引用)
    3. DAG结构信息(保持文献组织逻辑)
    """
    
    # 构建可用论文的详细信息
    papers_detail = "\n**可用的参考文献详细信息:**\n"
    for paper in papers:
        papers_detail += f"\n---\n"
        papers_detail += f"**Paper ID:** {paper.get('paper_id', 'N/A')}\n"
        papers_detail += f"**Title:** {paper.get('title', 'N/A')}\n"
        papers_detail += f"**Abstract:** {paper.get('abstract', 'N/A')[:300]}...\n"
        papers_detail += f"**Summary (关于源论文的总结):** {paper.get('summary', 'N/A')}\n"
        papers_detail += f"**Citation Format:** {paper.get('citation', 'N/A')}\n"
        # 注意:这里可以访问full_text_segments进行更准确的引用
    
    # 构建DAG结构指导
    dag_guidance = ""
    if dag_context and "dimensions" in dag_context:
        dag_guidance = f"\n**文献组织结构要求:**\n"
        dag_guidance += f"Related Work需要按照以下{len(dag_context['dimensions'])}个维度组织:\n"
        for i, dim in enumerate(dag_context['dimensions'], 1):
            grouped_papers = dag_context.get('grouped_papers', {}).get(dim, [])
            dag_guidance += f"\n{i}. **{dim}**\n"
            dag_guidance += f"   包含{len(grouped_papers)}篇相关论文:\n"
            for paper in grouped_papers[:5]:  # 每个维度最多显示5篇
                dag_guidance += f"   - {paper.get('title', 'N/A')} (ID: {paper.get('paper_id', 'N/A')})\n"
    
    # 构建反馈要点
    feedback_summary = "\n**反馈要点总结:**\n"
    if feedback_metadata:
        if "priority_issues" in feedback_metadata:
            feedback_summary += "**优先解决的问题:**\n"
            for issue in feedback_metadata.get("priority_issues", [])[:5]:
                feedback_summary += f"- {issue}\n"
    
    prompt = f"""
你是一位经验丰富的学术写作专家。请根据反馈意见修订以下Related Work部分。

**源论文摘要:**
{source_abstract}

{dag_guidance}

**原始Related Work:**
{related_work_original}

**收到的反馈:**
{feedback}

{feedback_summary}

{papers_detail}

**当前引用信息:**
共有{len(citations)}处引用,请在修订时确保:
1. 引用的paper_id与上述参考文献列表对应
2. 引用的内容准确反映被引论文的贡献
3. 可以使用上述论文的Summary来确保引用准确性

**修订要求:**

1. **结构调整:**
   - 严格遵循上述文献组织结构(各维度)
   - 每个维度内部遵循"问题→方法分类→局限性→创新点"的逻辑
   - 确保各部分过渡自然

2. **批判性分析强化:**
   - 在每类方法后,明确指出其局限性
   - 将局限性与源论文的创新点关联
   - 使用对比性语言突出源论文的优势

3. **引用准确性保证:**
   - 仅使用上述参考文献列表中的论文
   - 引用时参考提供的Summary,确保准确性
   - 保持原有的citation_text格式
   - 正确标注paper_id

4. **语言简洁化:**
   - 删除冗余描述
   - 合并相似的引用
   - 突出核心观点

5. **创新点对比明确:**
   - 在结论部分明确说明源论文的优势
   - 使用具体的对比说明为什么更好

**输出格式:**
你必须返回一个有效的JSON对象,格式如下:

{{
    "related_work": "修订后的完整Related Work文本...",
    "cite_ids": [
        {{"citation_text": "Smith et al., 2023", "paper_id": "1"}},
        {{"citation_text": "Jones & Brown, 2022", "paper_id": "2"}},
        ...
    ]
}}

**重要说明:**
- citation_text应该与related_work中的引用文本完全一致
- paper_id必须对应上述参考文献列表中的ID
- 按照引用在文本中出现的顺序记录cite_ids
- 同一篇论文被多次引用时,每次都要单独记录
- 不要包含任何markdown格式或额外说明
"""
    return prompt


def generate_enhanced_validation_prompt_with_full_text(
    claim: str,
    paper: Dict[str, Any],
    use_full_text: bool = True
) -> str:
    """
    增强版的验证prompt,使用完整的论文内容而不仅仅是summary
    
    Args:
        claim: 需要验证的声明
        paper: 论文的完整上下文(包括full_text_segments)
        use_full_text: 是否使用完整文本(如果为False则只用summary)
    """
    
    # 构建source内容
    source_content = f"**Title:** {paper.get('title', 'N/A')}\n\n"
    source_content += f"**Abstract:** {paper.get('abstract', 'N/A')}\n\n"
    
    if use_full_text and paper.get('full_text_segments'):
        source_content += "**Relevant Content from Paper:**\n"
        # 使用完整的页面内容,提高验证准确性
        for i, segment in enumerate(paper.get('full_text_segments', [])[:3], 1):
            source_content += f"\n--- Segment {i} ---\n{segment[:1000]}...\n"  # 限制每段长度
    else:
        source_content += f"**Summary:** {paper.get('summary', 'N/A')}\n"
    
    prompt = f"""
你是一位严谨的学术审稿专家。请判断以下声明(claim)是否被源文献(source)充分支持。

**声明(Claim):**
{claim}

**源文献(Source):**
{source_content}

**判断标准:**
- "Yes": 声明的内容在源文献中有明确且准确的支持
- "No": 声明存在以下任一情况:
  1. 直接矛盾(声明与源文献明确矛盾)
  2. 信息缺失(声明的内容在源文献中找不到支持)
  3. 误述/不精确(声明歪曲了源文献的原意或重点)
  4. 错误归因(声明将源文献的贡献归于其他实体)

请仅回答"Yes"或"No",不要包含任何其他解释或文字。
"""
    return prompt
