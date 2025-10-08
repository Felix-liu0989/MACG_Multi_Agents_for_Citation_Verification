import fitz
import os
import re
import json
import arxiv
from tqdm import tqdm
# from citegeist.utils.prompts import process_data_for_related_work_prompt

def download_pdf(paper_ids:list[str], dirpath:str):
    pdf_infos = []
    for paper_id in tqdm(paper_ids):
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id])))
        pdf_path = paper.download_pdf(dirpath=dirpath, filename=paper_id + ".pdf")
        text = pdf_to_text(pdf_path)
        pdf_infos.append({"id": paper_id, "text": text,"path": pdf_path})
    return pdf_infos



def pdf_to_text(pdf_path):
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def extract_after_related_work_regex(text):
    """使用正则表达式提取related work之后的内容"""
    
    # 匹配多种related work格式的正则表达式
    pattern = r'(?:^|\n)\s*(?:related\s+works?|literature\s+review|background)\s*:?\s*'
    
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    
    if match:
        # 从匹配位置之后开始提取
        start_pos = match.end()
        return text[start_pos:].strip()
    else:
        return None

def process_pdf(pdf_infos:list[dict]):
    for pdf_info in pdf_infos:
        text = pdf_info["text"]
        if "related work" in text.lower():
            text = extract_after_related_work_regex(text)
            if text is not None:
                pdf_info["text"] = "related work: " + text
        else:
            pdf_infos.remove(pdf_info)
    return pdf_infos

# roadmap: (1) 将pdf_infos中的text进行处理，提取related work之后的内容，并保存到pdf_infos中；
#          (2) 将related work中的reference提取出来, 保存title
#          (3) 与scholar_copilot_eval_data_1k_related_work.json进行匹配，如果匹配到，则保存title和abstract
#          (4) 检索abstract，返回相似度最高的3个论文id，并生成citation
#          (5) 根据id下载pdf, 提取text， 作为检索池

if __name__ == "__main__":
    paper_ids = [
        "2201.11903v1",
        "2402.10890v1",
        "1805.04833v1",
        "1904.09751v1",
        "2202.00666v1",
        "2309.05653v1"
    ]
    pdf_infos = download_pdf(paper_ids, "results/pdfs")
    
    print(pdf_infos[0])