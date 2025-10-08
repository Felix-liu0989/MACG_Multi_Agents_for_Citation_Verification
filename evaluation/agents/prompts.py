CRITERIA = {'Coverage': {'description': 'Coverage: Coverage assesses the extent to which the related work encapsulates all relevant aspects of the topic, ensuring comprehensive discussion on both central and peripheral topics.',
                         'score 1': 'The related work has very limited coverage, only touching on a small portion of the topic and lacking discussion on key areas.',
                         'score 2': 'The related work covers some parts of the topic but has noticeable omissions, with significant areas either underrepresented or missing.',
                         'score 3': 'The related work is generally comprehensive in coverage but still misses a few key points that are not fully discussed.',
                         'score 4': 'The related work covers most key areas of the topic comprehensively, with only very minor topics left out.',
                         'score 5': 'The related work comprehensively covers all key and peripheral topics, providing detailed discussions and extensive information.', },

            'Structure': {'description': 'Structure: Structure evaluates the logical organization and coherence of sections and subsections, ensuring that they are logically connected.',
                          'score 1': 'The related work lacks logic, with no clear connections between sections, making it difficult to understand the overall framework.',
                          'score 2': 'The related work has weak logical flow with some content arranged in a disordered or unreasonable manner.',
                          'score 3': 'The related work has a generally reasonable logical structure, with most content arranged orderly, though some links and transitions could be improved such as repeated subsections.',
                          'score 4': 'The related work has good logical consistency, with content well arranged and natural transitions, only slightly rigid in a few parts.',
                          'score 5': 'The related work is tightly structured and logically clear, with all sections and content arranged most reasonably, and transitions between adajecent sections smooth without redundancy.', },

            'Relevance': {'description': 'Relevance: Relevance measures how well the content of the related work aligns with the research topic and maintain a clear focus.',
                          'score 1': 'The  content is outdated or unrelated to the field it purports to review, offering no alignment with the topic',
                          'score 2': 'The related work is somewhat on topic but with several digressions; the core subject is evident but not consistently adhered to.',
                          'score 3': 'The related work is generally on topic, despite a few unrelated details.',
                          'score 4': 'The related work is mostly on topic and focused; the narrative has a consistent relevance to the core subject with infrequent digressions.',
                          'score 5': 'The related work is exceptionally focused and entirely on topic; the article is tightly centered on the subject, with every piece of information contributing to a comprehensive understanding of the topic.', }}


CRITERIA_BASED_JUDGING_PROMPT_TEXT_QUALITY = '''

You are an expert academic reviewer. Please evaluate the following "Related Work" section according to five key criteria. For each criterion, give a score from 0 (very poor) to 5 (excellent), and briefly justify your score in 1-2 sentences. Be strict in your evaluation, especially for "Comprehensiveness and Relevance"—note that the information source is limited to retrieved papers, so some relevant works may be missing.

**Criteria:**

1. **Clarity:** Is the writing easy to understand and precise? Does it avoid unnecessary jargon and redundancy? Is information conveyed efficiently? Redundancy citations are key to evaluate this criterion. Because the related work should be brief and concise. 
2. **Structure:** Is the overall structure logical? Are there clear transitions between ideas and paragraphs, creating a coherent and easy-to-follow narrative? **Are there clear and appropriate subsection titles that help the reader quickly grasp the content hierarchy?** If the related work does not have clear subsection titles, it should be given a low score.
3. **Relevance:** Does the section cover the most important and relevant literature in the field? Does it demonstrate a thorough understanding of the area, and are the selected works highly relevant to the current research?  
4. **Motivation:** Does the section effectively highlight the limitations, shortcomings, or unexplored areas in prior work? Does it clearly explain why the current research is necessary and what specific gap it aims to fill? If the section doesn't compare the existing works, it should be given a low score.
5. **Critics:** Does the section provide in-depth analysis and critique of the existing literature? Instead of merely listing prior works, does it identify the strengths, weaknesses, applicability, and limitations of different approaches? Does it demonstrate the author's independent thinking and critical perspective on the current state of the field?  

**Please output in the following format:**
```json
{
  "Clarity": {"score": X, "justification": "..."},
  "Structure": {"score": X, "justification": "..."},
  "Relevance": {"score": X, "justification": "..."},
  "Motivation": {"score": X, "justification": "..."},
  "Critics": {"score": X, "justification": "..."}
}
```
Here is an academic related work based on the given abstract:
---
Abstract: 
[ABSTRACT]
---
Related Work to evaluate:
[RELATED_WORK]
---
'''




CRITERIA_BASED_JUDGING_PROMPT_TEXT_QUALITY_BASELINES = '''
You are an expert academic reviewer. Please evaluate the following "Related Work" section according to five key criteria. For each criterion, give a score from 0 (very poor) to 5 (excellent), and briefly justify your score in 1-2 sentences. Be strict in your evaluation, especially for "Comprehensiveness and Relevance"—note that the information source is limited to retrieved papers, so some relevant works may be missing.

**Criteria:**

1. **Clarity:** Is the writing easy to understand and precise? Does it avoid unnecessary jargon and redundancy? Is information conveyed efficiently? igore redundancy citations in the related work.
2. **Structure:** Is the overall structure logical? Are there clear transitions between ideas and paragraphs, creating a coherent and easy-to-follow narrative? **Are there clear and appropriate subsection titles that help the reader quickly grasp the content hierarchy?**
3. **Relevance:** Does the section cover the most important and relevant literature in the field? Does it demonstrate a thorough understanding of the area, and are the selected works highly relevant to the current research?  **Please be strict when evaluating comprehensiveness, and note that due to the limitation of information sources (restricted to retrieved papers), some relevant works may be missing.**
4. **Motivation:** Does the section effectively highlight the limitations, shortcomings, or unexplored areas in prior work? Does it clearly explain why the current research is necessary and what specific gap it aims to fill?
5. **Critics:** Does the section provide in-depth analysis and critique of the existing literature? Instead of merely listing prior works, does it identify the strengths, weaknesses, applicability, and limitations of different approaches? Does it demonstrate the author's independent thinking and critical perspective on the current state of the field?

**Please output in the following format:**
```json
{
  "Clarity": {"score": X, "justification": "..."},
  "Structure": {"score": X, "justification": "..."},
  "Relevance": {"score": X, "justification": "..."},
  "Motivation": {"score": X, "justification": "..."},
  "Critics": {"score": X, "justification": "..."}
}
```
Here is an academic related work based on the given abstract:
---
Abstract: 
[ABSTRACT]
---
Related Work to evaluate:
[RELATED_WORK]
---


'''

CRITERIA_BASED_JUDGING_PROMPT  = '''
Here is an academic related work based on the given abstract "[ABSTRACT]":
---
[RELATED WORK]
---

<instruction>
Please evaluate this related work based on the criterion above provided below, and give a score from 1 to 5 according to the score description:
---
Criterion Description: [Criterion Description]
---
Score 1 Description: [Score 1 Description]
Score 2 Description: [Score 2 Description]
Score 3 Description: [Score 3 Description]
Score 4 Description: [Score 4 Description]
Score 5 Description: [Score 5 Description]
---
Return the score without any other information:
'''

NLI_PROMPT = '''
---
Claim:
[CLAIM]
---
Source: 
[SOURCE]
---
Claim:
[CLAIM]
---
Is the Claim faithful to the Source? 
A Claim is faithful to the Source if the core part in the Claim can be supported by the Source.\n
Only reply with 'Yes' or 'No':
At least one sentence in the Claim must be supported by the Source.
'''


CHECK_CITATION_PROMPT = '''
You are an expert in artificial intelligence who wants to write a overall and comprehensive related work based on the given abstract "[ABSTRACT]".\n\
Below are a list of papers for references:
---
[PAPER LIST]
---
You have written a subsection below:\n\
---
[SUBSECTION]
---
<instruction>
The sentences that are based on specific papers above are followed with the citation of "paper_title" in "[]".
For example 'the emergence of large language models (LLMs) [Language models are few-shot learners; Language models are unsupervised multitask learners; PaLM: Scaling language modeling with pathways]'

Here's a concise guideline for when to cite papers in a related work:
---
1. Summarizing Research: Cite sources when summarizing the existing literature.
2. Using Specific Concepts or Data: Provide citations when discussing specific theories, models, or data.
3. Comparing Findings: Cite relevant studies when comparing or contrasting different findings.
4. Highlighting Research Gaps: Cite previous research when pointing out gaps your related work addresses.
5. Using Established Methods: Cite the creators of methodologies you employ in your related work.
6. Supporting Arguments: Cite sources that back up your conclusions and arguments.
7. Suggesting Future Research: Reference studies related to proposed future research directions.
---

Now you need to check whether the citations of "paper_title" in this subsection is correct.
A correct citation means that, the content of corresponding paper can support the sentences you write.
Once the citation can not support the sentence you write, correct the paper_title in '[]' or just remove it.

Remember that you can only cite the 'paper_title' provided above!!!
Any other informations like authors are not allowed cited!!!
Do not change any other things except the citations!!!
</instruction>
Only return the subsection with correct citations:
'''

SUBSECTION_WRITING_PROMPT = '''
You are an expert in artificial intelligence who wants to write a overall and comprehensive related work based on the given abstract "[ABSTRACT]".\n\
You have created a overall outline below:\n\
---
[OVERALL OUTLINE]
---
Below are a list of papers for references:
---
[PAPER LIST]
---

<instruction>
Now you need to write the content for the subsection:
"[SUBSECTION NAME]" under the section: "[SECTION NAME]"
The details of what to write in this subsection called [SUBSECTION NAME] is in this descripition:
---
[DESCRIPTION]
---

Here is the requirement you must follow:
1. The content you write must be more than [WORD NUM] words.
2. When writing sentences that are based on specific papers above, you cite the "paper_title" in a '[]' format to support your content. An example of citation: 'the emergence of large language models (LLMs) [Language models are few-shot learners; PaLM: Scaling language modeling with pathways]'
    Note that the "paper_title" is not allowed to appear without a '[]' format. Once you mention the 'paper_title', it must be included in '[]'. Papers not existing above are not allowed to cite!!!
    Remember that you can only cite the paper provided above and only cite the "paper_title"!!!
3. Only when the main part of the paper support your claims, you cite it.


Here's a concise guideline for when to cite papers in a related work:
---
1. Summarizing Research: Cite sources when summarizing the existing literature.
2. Using Specific Concepts or Data: Provide citations when discussing specific theories, models, or data.
3. Comparing Findings: Cite relevant studies when comparing or contrasting different findings.
4. Highlighting Research Gaps: Cite previous research when pointing out gaps your related work addresses.
5. Using Established Methods: Cite the creators of methodologies you employ in your related work.
6. Supporting Arguments: Cite sources that back up your conclusions and arguments.
7. Suggesting Future Research: Reference studies related to proposed future research directions.
---

</instruction>
Return the content of subsection "[SUBSECTION NAME]" in the format:
<format>
[CONTENT OF SUBSECTION]
</format>
Only return the content more than [WORD NUM] words you write for the subsection [SUBSECTION NAME] without any other information:
'''


LCE_PROMPT = '''
You are an expert in artificial intelligence who wants to write a overall and comprehensive related work based on the given abstract "[ABSTRACT]".

Now you need to help to refine one of the subsection to improve th ecoherence of your related work.

You are provied with the content of the subsection along with the previous subsections and following subsections.

Previous Subsection:
--- 
[PREVIOUS]
---

Following Subsection:
---
[FOLLOWING]
---

Subsection to Refine: 
---
[SUBSECTION]
---


If the content of Previous Subsection is empty, it means that the subsection to refine is the first subsection.
If the content of Following Subsection is empty, it means that the subsection to refine is the last subsection.

Now refine the subsection to enhance coherence, and ensure that the content of the subsection flow more smoothly with the previous and following subsections. 

Remember that keep all the essence and core information of the subsection intact. Do not modify any citations in [] following the sentences.

Only return the whole refined content of the subsection without any other informations (like "Here is the refined subsection:")!

The subsection content:
'''

LCE_PROMPT_LATEX = '''
You are an expert in artificial intelligence who wants to write a overall and comprehensive related work based on the given abstract "[ABSTRACT]".

Now you need to help to refine one of the subsections to improve the coherence of your related work.

You are provided with the content of the subsection along with the previous subsections and following subsections.

Previous Subsection:
--- 
[PREVIOUS]
---

Following Subsection:
---
[FOLLOWING]
---

Subsection to Refine: 
---
[SUBSECTION]
---

If the content of the Previous Subsection is empty, it means that the subsection to refine is the first subsection.
If the content of the Following Subsection is empty, it means that the subsection to refine is the last subsection.

Now refine the subsection to enhance coherence, and ensure that the content of the subsection flows more smoothly with the previous and following subsections. 

Remember to keep all the essence and core information of the subsection intact. Do not modify any citations in [] following the sentences.

Convert the refined content into LaTeX format to ensure it is properly formatted for a LaTeX document. Include any necessary LaTeX commands for sections, subsections, equations, figures, tables, and other elements as needed.

Only return the whole refined content of the subsection in LaTeX format without any other information (like "Here is the refined subsection:")!

The subsection content:
'''

OUTLINE_EVALUATION_PROMPT = """
[Task]
Rigorously evaluate the quality of an academic related work outline based on the given abstract [ABSTRACT] by scoring three dimensions (each 0-100) and calculating the average as the final score. 

[Evaluation Criteria]  
Evaluate each dimension on a 0-5 scale based strictly on the highest standards below. The final score is the average of the three dimension scores.

1. **Structural Coherence & Narrative Logic** (5 points):  
   - **Ideal Standard**: The outline demonstrates a clear and logical flow, with sections and subsections organized to guide the reader effectively. Transitions are smooth, and the structure supports a coherent narrative.  
   - **Scoring**: Deduct points for issues like imbalanced section lengths, weak transitions, or subsections that disrupt the narrative flow. A perfect score (10) requires no observable flaws.

2. **Conceptual Depth & Thematic Coverage** (5 points):  
   - **Ideal Standard**: The outline comprehensively covers key themes, concepts, and subtopics, balancing depth and breadth. It reflects a mastery of the field’s core debates and evolution.  
   - **Scoring**: Deduct points for missing critical themes, overemphasizing niche areas, or superficial treatment of foundational theories.

3. **Critical Thinking & Scholarly Synthesis** (5 points):  
   - **Ideal Standard**: The outline critically analyzes literature gaps, methodological conflicts, and scholarly disagreements. It synthesizes perspectives to reveal insights beyond mere summary.  
   - **Scoring**: Deduct points for lacking analysis of contradictions, ignoring major critiques, or failing to propose unresolved questions.

[Abstract]
[ABSTRACT]

[Skeleton]
[OUTLINE]

[Output Format]
Rationale:
<Provide a detailed reason for the score, considering all dimensions step by step. Highlight specific strengths and weaknesses, such as structural imbalances, thematic gaps, or insufficient critical analysis. Then provide the final scores for each dimension>
- Structure: <X/5>  
- Coverage: <Y/5>  
- Critical Analysis: <Z/5>  

Final Score: 
<SCORE>(X+Y+Z)/3</SCORE>  
(Example: <SCORE>2.3</SCORE>; scores can include two decimal place)

"""




LANGUAGE_EVALUATION_PROMPT = """

[Task]
Rigorously evaluate the quality of an academic related work based on the given abstract [ABSTRACT] by scoring three dimensions (each 0-5) and calculating the average as the final score. 

[Evaluation Criteria]  
Evaluate each dimension on a 0-5 scale based strictly on the highest standards below. The final score is the average of the three dimension scores.

1. **Academic Formality** (5 points):  
   - Demonstrates *flawless* academic rigor. Uses precise terminology consistently, avoids colloquial language entirely, and maintains a strictly scholarly tone. Sentence structures are sophisticated and purposefully crafted to enhance analytical depth. **Even a single instance of informal phrasing or imprecise terminology disqualifies a perfect score**.
2. **Clarity & Readability** (5 points):  
   - Writing is *exceptionally* clear and concise. Sentences are logically structured, with no ambiguity. Transitions between ideas are seamless, and the argument progresses with precision. **Any unnecessary complexity or minor ambiguity precludes full marks.**  
3. **Redundancy** (5 points):  
   - **Unique**: each sentence must have unique value and cannot be repeated. Repetition is only allowed to maintain structural coherence, such as using uniform terminology or necessary transitional phrases. Repeating key concept definitions in a new context to help readers understand can be seen as a structural requirement.
   - **Efficient argumentation**: Argumentation needs to be efficient, with logically coherent viewpoints and avoiding unnecessary repetition. Even minor repetitions without actual structural effects can result in deduction of points. For example, repeating a discovery almost identical in the same paragraph without providing new insights or perspectives will result in deduction of points. 

[Abstract]
[ABSTRACT]

[Section]
[SECTION]

[Output Format]

<SCORE>(X+Y+Z)/3</SCORE>  
(Example: <SCORE>2.3</SCORE>; scores can include two decimal place)
"""


CRITICAL_EVALUATION_PROMPT = """
[Task]
Rigorously evaluate the quality of an academic related work based on the given abstract [ABSTRACT] by scoring three dimensions (each 0-5) and calculating the average as the final score. 
[Evaluation Criteria]  
The final score is the sum of the individual scores from the following three dimensions. Please evaluate each dimension thoroughly and rigorously.

1. **Critical Analysis** (5 points):  
   - Offers a deep, incisive critique of methodologies, results, and underlying assumptions. Provides a clear identification of significant gaps, weaknesses, and areas for improvement. Challenges assumptions with well-supported arguments, offering clear alternatives or improvements.  

2. **Original Insights** (5 points):  
   - Proposes novel, well-supported interpretations or frameworks based on the reviewed literature. Demonstrates a strong understanding of the subject matter and provides genuinely original contributions that challenge the status quo. Insights are clearly connected to existing research, offering fresh perspectives or unique ways forward.  

3. **Future Directions** (5 points):  
   - Clearly identifies specific, promising research directions with strong justification. Suggests actionable, concrete ideas for future research that are rooted in the gaps identified within the reviewed literature. Demonstrates foresight in proposing innovative approaches and methodologies.  

[Abstract]
[ABSTRACT]

[Section]
[SECTION]

[Output Format]
Rationale:
<Provide a detailed reason for the score, considering all dimensions step by step. Highlight specific strengths and weaknesses, such as the depth of critique, the originality of insights, or the clarity of future directions.>
Final Score: 
<SCORE>(X+Y+Z)/3</SCORE>  
(Example: <SCORE>2.3</SCORE>; scores can include two decimal place)
"""

CRITICAL_EVALUATION_PROMPT_Two_Related_Works_Compared_CRITERIA = """
[Task]
Evaluate the quality of two academic related work based on the given abstract [ABSTRACT] by scoring three dimensions (each 0-5) and calculating the average as the final score. 
[Evaluation Criteria]  
The final score is the sum of the individual scores from the following three dimensions. Please evaluate each dimension thoroughly and rigorously.

# Concise Evaluation Prompt for Conference Paper Related Work  
(Aligned with Critical Analysis, Original Insights, Future Directions; 15 points total)  


## 1. Evaluation Premise (Conference Context)  
Assess against top CS/AI conference (NeurIPS/ICML/ACL) requirements:  
- Length: ≤1.5 single-column pages (concise info delivery).  
- Audience: Domain researchers (no over-explaining basic methods).  
- Core goal: Rapidly link "literature status → key gaps → own work’s novelty" (not exhaustive review).  


## 2. Dimension-Specific Criteria  

### 1. Critical Analysis (5 pts)  
- (2 pts) Critiques **specific limitations** of core methods (not vague claims, e.g., "DOTS focuses on single optimal paths, ignoring diversity" vs. "lacks diversity").  
- (2 pts) Supports critiques with **1-2 literature examples** (e.g., "Proof Flow (Ho et al., 2024) needs large reward datasets, limiting efficiency").  
- (1 pt) Distills a **central domain gap** (e.g., "no method balances diversity-quality efficiently in complex tasks") from scattered limitations.  


### 2. Original Insights (5 pts)  
- (2 pts) Clarifies **differentiation from existing work** (e.g., "our training-time optimization contrasts with DOTS’ inference-time search").  
- (2 pts) Proposes **novel, targeted solutions** to gaps (e.g., "modeling reasoning as DAGs (vs. sequences) for multi-path sampling").  
- (1 pt) Links insights directly to identified gaps (e.g., "DAG sampling addresses ‘sequential methods miss diverse paths’").  


### 3. Future Directions (5 pts)  
- (2 pts) Suggests **specific, actionable ideas** tied to gaps (e.g., "extend DAG sampling to multimodal tasks with cross-modal rewards").  
- (2 pts) Ties directions to method limitations (e.g., "explore unsupervised reward modeling to fix GFlowNets’ data dependence").  
- (1 pt) Focuses on "small, precise" directions (not broad claims like "study diversity more").  


## 3. Overall Judgment Rules  
1. Priority: Info efficiency (gap/own work clarity) > detail depth > literature exhaustiveness.  
2. Deduction: Max 2 pts per dimension if not conference-aligned (e.g., vague critiques); 0 pts if a dimension is missing.  
3. High-score (≥12 pts): No major weaknesses + at least 1 dimension (e.g., insights) highlights conference-specific focus.


[Abstract]
[ABSTRACT]

[Related Work 1]
[RELATED_WORK1]

[Related Work 2]
[RELATED_WORK2]

[Output Format]

Rationale:
<Provide a brief reason for the score, considering all dimensions step by step. Highlight specific strengths and weaknesses, such as the depth of critique, the originality of insights, or the clarity of future directions.>
Final Score: 
<SCORE>(X+Y+Z)/3</SCORE>  
(Example: <SCORE>2.3</SCORE>; scores can include two decimal place)
return the result in the format of json:
{{
   "work1_score": <SCORE>(X+Y+Z)/3</SCORE>,
   "work2_score": <SCORE>(X+Y+Z)/3</SCORE>,
   "winner": <"work1" or "work2">
   "rationale": <Provide a detailed reason for the score, considering all dimensions step by step. Highlight specific strengths and weaknesses, such as the depth of critique, the originality of insights, or the clarity of future directions.>
}}
"""

CRITICAL_EVALUATION_PROMPT_OUTLINES = """
[Task]
Evaluate the quality of an academic related work outline based on the given abstract [ABSTRACT] by scoring three dimensions (each 0-5) and calculating the average as the final score. 
1. 目标文章基础信息：标题【[TITLE]】、摘要【[ABSTRACT]】
2. 待评估的文献组织结果：{此处填入生成的文献组织结构，如“DAG子节点/大纲小节名称+对应文献列表（每篇文献含标题+核心观点）”}

评估规则（严格参考文件对Related Work的要求）：
- 5分：所有文献组（DAG子节点/大纲小节）的核心讨论均与目标文章核心主题（从摘要中提取的核心研究方向，如“多步推理的DAG结构化组织”“related work的双模型事实核查”等）完全匹配，能直接为目标文章的创新点提供背景支撑；
- 4分：90%以上文献组的核心讨论与目标文章核心主题匹配，仅极个别文献组存在轻微偏离（不影响整体关联性）；
- 3分：70%-89%文献组的核心讨论与目标文章核心主题匹配，部分文献组讨论方向与核心主题关联较弱；
- 2分：50%-69%文献组的核心讨论与目标文章核心主题匹配，超过1/3文献组讨论方向偏离核心主题；
- 1分：30%-49%文献组的核心讨论与目标文章核心主题匹配，多数文献组讨论方向与核心主题无关；
- 0分：低于30%文献组的核心讨论与目标文章核心主题匹配，整体文献组织与目标文章脱节。

输出要求：
1. 首先明确目标文章的核心主题（基于输入的摘要提取，需与文件中“Related Work需contextualize目标文章”的逻辑一致）；
2. 逐文献组分析其与核心主题的匹配情况（简要说明匹配/偏离理由）；
3. 给出最终TTC评分（0-5分），并总结整体契合度特征
"""




def get_citation_extraction_prompt(text: str) -> str:
    return f"""
    Analyze the following text and extract the citation in the format of "Author_Year".

    [Output Format]
    Strict one citation in the format of "Author_Year":
    
    {text}
    """

def get_extraction_prompt(text: str) -> str:
    """
    Generate optimized prompt for claim extraction (v2)
    """
    return f"""Analyze the following text and decompose it into independent claims following strict consolidation rules:

[Claim Definition]
A verifiable objective factual statement that functions as an independent knowledge unit. Each claim must:
1. Contain complete subject-predicate-object structure
2. Exist independently without contextual dependency
3. Exclude subjective evaluations

[Merge Rules]→ Should merge when:
- Same subject + same predicate + different objects (e.g., "Should measure A / Should measure B" → "Should measure A and B")
- Different expressions of the same research conclusion
- Parallel elements of the same category (e.g., "A, B and C")

[Separation Rules]→ Should keep separate when:
- Different research subjects/objects
- Claims with causal/conditional relationships
- Findings across temporal sequences
- Conclusions using different verification methods

[Output Format]
Strict numbered list with consolidated claims maintaining grammatical integrity:
1. Use "and/or/including" for merged items
2. Separate parallel elements with commas
3. Prohibit abbreviations or contextual references

Below is the text you need to extract claims from:

{text}
"""

def get_deduplication_prompt(facts_list: list) -> str:
    """
    生成用于对 facts 列表去重的 Prompt。要求输出需要删除的序号（逗号分隔）。
    """
    numbered_facts = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(facts_list)])
    return f"""Below is a numbered list of claims. Your task is to identify and group 
claims that convey the same information, removing all redundancy.

[Guidelines]:
- Claims that express the same fact or knowledge in different wording or detail are duplicates.
- If one claim is fully included within another or repeats the same idea, consider it a duplicate.
- Claims with differing details, context, or scope are not duplicates.

For each group of duplicates, output the serial numbers of the claims to be removed (comma-separated). 
Choose one claim to keep.

Example:
If claims 2, 5, and 8 are duplicates and claim 2 is kept, output "5,8".

List of claims:
{numbered_facts}

Output ONLY the serial numbers to remove. No additional text.
"""