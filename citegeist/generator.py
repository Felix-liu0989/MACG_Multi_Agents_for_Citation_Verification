# Imports
import json
import math
import os
import sys
import re
import transformers
sys.path.append(".")
from evaluation.agents.judge import Judge
from typing import Callable, Optional
import datetime
from multi_dims.model_definitions import initializeLLM, promptLLM, constructPrompt
from multi_dims.pipeline import run_dag_to_classifier
from bertopic import BERTopic
from dotenv import load_dotenv
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from multi_dims.paper import Paper
from multi_dims.visualizer import visualize_dags
from multi_dims.builder import build_dags, update_roots_with_labels
from multi_dims.classifier import label_papers_by_topic
from .utils.citations import (
    filter_citations,
    get_arxiv_abstract,
    get_arxiv_citation,
    get_arxiv_title,
    process_arxiv_paper_with_embeddings,
)
from .utils.filtering import (
    select_diverse_pages_for_top_b_papers,
    select_diverse_papers_with_weighted_similarity,
)
from .utils.llm_clients import create_client
from .utils.llm_clients.deepseek_client import DeepSeekClient
from .utils.llm_clients.deepseek_client_multi_turn import DeepSeekClientMultiTurn
from .utils.prompts import (
    generate_brief_topic_prompt,
    generate_question_answer_prompt,
    generate_related_work_prompt,
    generate_summary_prompt_question_with_page_content,
    generate_summary_prompt_with_page_content,
    generate_related_work_outline_prompt,
    generate_related_work_prompt_with_arxiv_trees,
    generate_related_work_revision_prompt,
    generate_original_related_work_feedback_prompt,
    generate_related_work_outline_prompt_various_1,
    generate_related_work_revision_prompt_without_DAG,
    generate_summary_prompt_with_second_round_retrieved_content,
    process_data_for_extract_cited_sentences,
    process_data_for_classify_errors,
    process_data_for_correct_citation_errors,
    generate_related_work_revise_citations_format
)
import json_repair
from pathlib import Path
from .utils.agent_context import (
    AgentCommunicationContext,
    PaperContext,
    DAGContext,
    CitationContext
)
from .utils.prompts_enhanced import (
    generate_enhanced_feedback_prompt,
    generate_enhanced_revision_prompt
)
import time
# Load environment variables
load_dotenv()



class Args:
    def __init__(self):
        self.topic = "topic"
        self.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods"]
        self.llm = 'gpt'
        self.init_levels = 2

        self.dataset = "Reasoning"
        self.data_dir = f"datasets/multi_dim/{self.dataset.lower().replace(' ', '_')}/"
        self.internal = f"{self.dataset}.txt"
        self.external = f"{self.dataset}_external.txt"
        self.groundtruth = "groundtruth.txt"
        self.max_density = 5   
        self.max_depth   = 3
        self.length = 512
        self.dim = 768
        self.iters = 4
global args
args = Args()
args = initializeLLM(args)

args1 = Args()
args1 = initializeLLM(args1)

args2 = Args()
args2 = initializeLLM(args2)


judge_gemini = Judge(model="google/gemini-2.5-flash")
judge_deepseek = Judge(model="deepseek-chat")

class Generator:
    """Main generator class for Citegeist."""

    def __init__(
        self,
        llm_provider: str,
        database_uri: str,  # path to local milvus DB file or remote hosted Milvus DB
        database_token: Optional[str] = None,  # This only has to be set when authentication is required for the DB
        sentence_embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        topic_model_name: str = "Ah7med/BERTopic_ArXiv",
        **llm_kwargs,
    ):
        """
        Initialize the Generator with configuration.

        Args:
            llm_provider: LLM provider name ('azure', 'openai', 'anthropic').
                Falls back to environment variable LLM_PROVIDER, then to 'azure'
            sentence_embedding_model_name: Name of the sentence transformer embedding model
            topic_model_name: Name of the BERTopic model
            database_uri: Path to the Milvus database
            database_token: Optional token for accessing Milvus database
            **llm_kwargs: Provider-specific configuration arguments for the LLM client
        """
        # Initialize core models
        self.topic_model = BERTopic.load(topic_model_name,embedding_model=sentence_embedding_model_name)
        self.sentence_embedding_model = SentenceTransformer(sentence_embedding_model_name)
        if database_token is None:
            self.db_client = MilvusClient(uri=database_uri)
        else:
            self.db_client = MilvusClient(uri=database_uri, token=database_token)

        # Set up LLM client
        self.llm_provider = llm_provider

        # Create LLM client (falls back to value of LLM_PROVIDER in env variables, and finally falls back to azure)
        self.llm_client = create_client(self.llm_provider, **llm_kwargs)
        chat_tokenizer_dir = r"D:\Mydesktop\CitAgent\Code\Supplementary material and code\Multi_Agents_for_Citation_Verification\citegeist\tokenizer"
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(chat_tokenizer_dir,trust_remote_code=True)
        # Store API version for Azure compatibility
        self.api_version = os.getenv("AZURE_API_VERSION", "2023-05-15")
    def test_args(self):
        args.topic = "natural language processing"
        return args.topic
    
    def _count_tokens(self,text) -> int:
        return len(self.tokenizer.encode(text))
    
    
    def generate_related_work_MACG_optimized(
            self,
            abstract: str,
            breadth: int,
            depth: int,
            diversity: float,
            arxivid: str,
            status_callback: Optional[Callable] = None,
    ):
        if status_callback:
            status_callback(1, "Initializing")
        print("摘要：", abstract)
        print("向量化摘要并在milvus库中检索...")
        
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
        
        # 向量化摘要并在milvus库中检索  
        embedded_abstract = self.sentence_embedding_model.encode(abstract)
        topic = self.topic_model.transform(abstract)
        topic_id = topic[0][0]
        
        # 检索milvus库中与摘要最相似的论文
        if status_callback:
            status_callback(2, "Querying Vector DB for matches (this may take a while)")
            
        query_data: list[list[dict]] = self.db_client.search(
            collection_name = "abstracts",
            data = [embedded_abstract],
            limit = 6 * breadth,
            anns_field = "embedding",
            search_params = {"metric_type": "COSINE", "params": {}},
            output_fields = ["embedding"],
        )
        
        if status_callback:
            status_callback(3, f"Retrieved {len(query_data[0])} papers from the DB")
            
        # 清理DB响应数据
        papers_data: list[dict] = query_data[0]
        for obj in papers_data:
            obj["embedding"] = obj["entity"]["embedding"]
            obj.pop("entity")
            
        # 选择一个长列表的论文
        selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
            paper_data = papers_data,
            k = 3 * breadth,
            diversity_weight = diversity)
        
        if status_callback:
            status_callback(4, f"Selected {len(selected_papers)} papers for the longlist, retrieving full text(s) (this might take a while)")
            
        # 生成每个论文的页面嵌入
        page_embeddings: list[list[dict]] = []
        for paper in selected_papers:
            arxiv_id = paper["id"]
            result = process_arxiv_paper_with_embeddings(arxiv_id, self.topic_model)
            if result:
                page_embeddings.append(result)
                
        if status_callback:
            status_callback(5, f"Generated page embeddings for {len(page_embeddings)} papers")
            
        # 生成一个短列表的论文（最多k页每篇，最多b篇）
        relevant_pages: list[dict] = select_diverse_pages_for_top_b_papers(
            paper_embeddings=page_embeddings,
            input_string=abstract,
            topic_model=self.topic_model,
            k=depth,
            b=breadth,
            diversity_weight=diversity,
            skip_first=False,
        )
        start_time = time.time()
        # with open("relevant_pages.json", "w", encoding="utf-8") as f:
        #     json.dump(relevant_pages, f, indent=4, ensure_ascii=False)

        if status_callback:
            status_callback(6, f"Selected {len(relevant_pages)} papers for the shortlist")
        
        selected_papers = []
        internal_collection = {}
        data = []
        id = 0
        start_time_summarization = time.time()
        # Generate summaries for individual papers (taking all relevant pages into account)
        for obj in relevant_pages[1:]:
            # Because paper_id != arXiv_id -> retrieve arXiv id/
            arxiv_id = papers_data[obj["paper_id"]]["id"]
            arxiv_abstract = get_arxiv_abstract(arxiv_id)
            text_segments = obj["text"]
            obj["cite_ids"] = [id]
            title = get_arxiv_title(arxiv_id)
            
            # Create prompt
            prompt = generate_summary_prompt_with_page_content(
                abstract_source_paper=abstract,
                abstract_to_be_cited=arxiv_abstract,
                page_text_to_be_cited=text_segments,
                sentence_count=5,
            )
            
            internal_collection[id] = Paper(
                id, 
                title, 
                arxiv_abstract, 
                label_opts=["tasks", "datasets", "methodologies", "evaluation_methods"], 
                internal=True
            )
            temp_dict = {"Title": title, "Abstract": arxiv_abstract}
            data.append(temp_dict)            
            # Use the appropriate LLM client based on the provider
            response: str = self.llm_client.get_completion(prompt)
            obj["summary"] = response
            
            obj["citation"] = get_arxiv_citation(arxiv_id)
            internal_collection[id].summary = response
            internal_collection[id].citations = get_arxiv_citation(arxiv_id)
            idx = {
                "cite_ids":[id],
                "title":title,
                "abstract":arxiv_abstract,
                "summary":response,
                "arxiv_id":arxiv_id,
                "citations":get_arxiv_citation(arxiv_id)
            }
            selected_papers.append(idx)
            
            
            # 创建PaperContext，保存所有信息
            paper_ctx = PaperContext(
                paper_id=id,
                arxiv_id=arxiv_id,
                title=title,
                abstract=arxiv_abstract,
                summary=response,
                citation=get_arxiv_citation(arxiv_id),
                full_text_segments=text_segments,
                cite_ids=[id]
            )
            context.papers.append(paper_ctx)
            id += 1
            
            
        # with open("internal_collection.json", "w", encoding="utf-8") as f:
        #     json.dump(data, f, indent=4, ensure_ascii=False)
        end_time_summarization = time.time()
        
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        results_dir = PROJECT_ROOT / f"selected_papers/{arxivid.replace('.','_')}"
        

        # 确保目录存在
        results_dir.mkdir(exist_ok=True,parents=True)
        with open(f"{results_dir}/{breadth}_{depth}_{diversity}.json", "w", encoding="utf-8") as f:
            json.dump(selected_papers, f, indent=4, ensure_ascii=False)
        
        # with open("/home/liujian/internal_collection.json", "r", encoding="utf-8") as f:
        #       internal_collection = json.load(f)
        if status_callback:
            status_callback(7, "Generated summaries of papers (and their pages)")
        
        start_time_DAG = time.time()
        # 生成本篇文章的topic
        topic_prompt = generate_brief_topic_prompt(abstract)
        topic = self.llm_client.get_completion(topic_prompt)
        print("topic:")
        print(topic)
        
        # 生成文献树
        args.topic = topic
        
        roots,dags = run_dag_to_classifier(
            args,
            internal_collection
        )
        
        print("正在生成可视化...")
        
        proj_root = Path(__file__).parent.parent.parent
        dir = str(proj_root / "multi_dim_literature_visualizations")
        visualizer,arxiv_trees = visualize_dags(roots, dags, output_dir=dir,topic=args.topic)

        ## 为生成related work，先生成大纲
        print("正在生成related work大纲...")
        outline_prompt = generate_related_work_outline_prompt_various_1(abstract,arxiv_trees)
        outline = self.llm_client.get_completion(outline_prompt)
        outline = json_repair.loads(outline)
        

        print("outline:")
        print(outline)
        
        subsection_titles = outline["outline"]
            
        
        args.dimensions = subsection_titles
        roots,dags,id2node,label2node = build_dags(args)
        results = label_papers_by_topic(
            args, 
            internal_collection,
            subsection_titles
        )
        print(results)
        
        update_roots_with_labels(roots, results, internal_collection, args)
        grouped = {dim:[
        {
            "paper_id":pid,
            "title":paper.title,
            "abstract":paper.abstract,
            "summary":paper.summary,
            "citations":paper.citations
        }
        for pid,paper in roots[dim].papers.items()
    ] for dim in args.dimensions}
        
        context.dag_context = DAGContext(
            dimensions=args.dimensions,
            grouped_papers=grouped,
            dag_structure={
                "roots": {dim: root.to_dict() for dim, root in roots.items()},
                "id2node": id2node,
                "label2node": label2node
            },
            topic=topic,
            outline=outline
        )
        dim_1 = args.dimensions[0]
        dim_2 = args.dimensions[1]
        grouped_dim_1 = grouped[dim_1]
        grouped_dim_2 = grouped[dim_2]
        
        
        end_time_DAG = time.time()
        
        # 记录agent执行
        context.add_agent_execution(
            agent_name="dag_builder",
            input_data={"dimensions": args.dimensions},
            output_data={"grouped_papers_count": sum(len(v) for v in grouped.values())},
            execution_time=time.time() - start_time_DAG
        )
        
        start_time_related_work = time.time()
        # 生成related work
        # 获取agent所需的完整上下文
        rw_context = context.get_context_for_agent("related_work_generator")
        prompt = generate_related_work_prompt_with_arxiv_trees(
            source_abstract=context.source_paper_abstract,
            dimensions=context.dag_context.dimensions,
            grouped=context.dag_context.grouped_papers
        )
        related_work_with_citations = self.llm_client.get_completion(prompt)
        related_work_with_citations = related_work_with_citations.replace("```json","").replace("```","")
        related_work_with_citations = json_repair.loads(related_work_with_citations)
        print("related_work_with_citations:")
        print(related_work_with_citations)
        
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
        related_work = related_work_with_citations["related_work"]
        citations = related_work_with_citations["cite_ids"]
        print("related_work:")
        print(related_work)
        print("citations:")
        print(citations)
        end_time_related_work = time.time()
        
        start_time_extract_cited_sentences = time.time()
        # prompt_for_extract_cited_sentences = process_data_for_extract_cited_sentences(related_work)
        # cited_sentences = self.llm_client.get_completion(prompt_for_extract_cited_sentences)
        # cited_sentences = json_repair.loads(cited_sentences)
        # print("cited_sentences:")
        # print(cited_sentences)
        
        # judge_gemini = Judge(model="google/gemini-2.5-flash")
        # judge_deepseek = Judge(model="deepseek-chat")
        
        # ids = [i for i in citations if "paper_id" in i and i["paper_id"] is not None]
        
        # selected_papers = relevant_pages
        
        client = DeepSeekClient(
            api_key = os.environ.get("DEEPSEEK_API_KEY", ""),
            model_name = "deepseek-chat"
        )
        start_time_feedback = time.time()
        
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
        print("feedback:")
        print(feedback)
        
        context.add_agent_execution(
            agent_name="feedback_agent",
            input_data={"related_work_length": len(context.related_work_original)},
            output_data={"feedback_length": len(feedback)},
            execution_time=time.time() - start_time_feedback
        )
        
        prompt_for_revision = generate_related_work_revision_prompt(abstract,related_work,feedback,citations,args.dimensions)
        related_work_revision = client.get_completion(prompt_for_revision)
        print("related_work_revision:")
        print(related_work_revision)
        related_work_revision = related_work_revision.replace("```json","").replace("```","")
        related_work_revision_dict = json_repair.loads(related_work_revision)
        related_work_revision = related_work_revision_dict["related_work"]
        citations = related_work_revision_dict["cite_ids"]
        end_time_feedback = time.time()
        
        related_work_revision,quotes_with_citation_info,validation_results,error_types = self.validate_and_correct_citations(related_work_revision,selected_papers,citations,arxiv_id)
        end_time = time.time()
        filtered_citations: list[str] = filter_citations(
            related_works_section=related_work, citation_strings=[obj["citation"] for obj in relevant_pages[1:]]
        )
        
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        results_dir = PROJECT_ROOT / f"results/{arxivid.replace('.','_')}"
        # 保存
        results_dir.mkdir(exist_ok=True,parents=True)
        results_dict = {
            "related_work": related_work_revision,
            "citations": citations,
            "quotes_with_citation_info": quotes_with_citation_info,
            "validation_results": validation_results,
            "error_types": error_types
        }
        with open(results_dir / f"results_{breadth}_{depth}_{diversity}.json", "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)
        print("filtered_citations:")
        print(filtered_citations)
        date = datetime.date.today()
        # os.makedirs(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}", exist_ok=True)
        # with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/related_work_with_citations.txt", "w", encoding="utf-8") as f:
        #     f.write("="*50 + "\n" + "related_work" + "\n" + "="*50 + "\n" + related_work + "\n" + "="*50 + "\n" + "citations" + "\n" + "="*50 + "\n" + str(filtered_citations))
        # with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/grouped_dim_1.json", "w", encoding="utf-8") as f:
        #     json.dump(grouped_dim_1,f,ensure_ascii=False,indent=4)
        # with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/grouped_dim_2.json", "w", encoding="utf-8") as f:
        #     json.dump(grouped_dim_2,f,ensure_ascii=False,indent=4)
        # with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/related_work_revision.txt", "w", encoding="utf-8") as f:
        #     f.write("="*50 + "\n" + "related_work_revision" + "\n" + "="*50 + "\n" + related_work_revision + "\n" + "="*50 + "\n" + "citations" + "\n" + "="*50 + "\n" + str(filtered_citations))
        if status_callback:
            status_callback(8, f"Generated related work section with {len(filtered_citations)} citations")
        args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods"]
        
        
        dag_time = end_time_DAG - start_time_DAG
        related_work_time = end_time_related_work - start_time_related_work
        feedback_time = end_time_feedback - start_time_feedback
        summarization_time = end_time_summarization - start_time_summarization
        all_time = end_time - start_time
        print("Time taken for summarization:", round(summarization_time, 2), "seconds")
        print("Time taken for DAG:", round(dag_time, 2), "seconds")
        print("Time taken for related work:", round(related_work_time, 2), "seconds")
        print("Time taken for feedback:", round(feedback_time, 2), "seconds")
        print("Time taken for all process:", round(all_time, 2), "seconds")
        time_dict = {
            "summarization_time": summarization_time,
            "dag_time": dag_time,
            "related_work_time": related_work_time,
            "feedback_time": feedback_time,
            "all_time": all_time
        }   
        final= {"related_works": related_work_revision, "citations": filtered_citations, "selected_papers": relevant_pages,"time_dict": time_dict}
        # with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/final.json", "w", encoding="utf-8") as f:
        #     json.dump(final,f,ensure_ascii=False,indent=4)
            
        args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods"]
        
        
        return final
    
    def generate_related_work_MACG_FAST(
            self,
            target_abstract: str,
            breadth: int,
            depth: int,
            diversity: float,
            arxivid: str,
            selected_papers: list[dict],
            status_callback: Optional[Callable] = None,
    ):
        start_time = time.time()
        internal_collection = {}
        for paper in selected_papers:
            idx = paper["cite_ids"][0]
            title = paper["title"]
            abstract = paper["abstract"]
            response = paper["summary"]
            arxiv_id = paper["arxiv_id"]
            citations = paper["citation"]
            internal_collection[idx] = Paper(
                idx, 
                title, 
                abstract, 
                label_opts=["tasks", "datasets", "methodologies", "evaluation_methods"], 
                internal=True)
            
            internal_collection[idx].summary = response
            internal_collection[idx].citations = get_arxiv_citation(arxiv_id)
        
        start_time_DAG = time.time()
        # 生成本篇文章的topic
        topic_prompt = generate_brief_topic_prompt(target_abstract)
        topic = self.llm_client.get_completion(topic_prompt)
        print("topic:")
        print(topic)
        
        # 生成文献树
        args.topic = topic
        
        roots,dags = run_dag_to_classifier(
            args,
            internal_collection
        )
        
        print("正在生成可视化...")
        
        proj_root = Path(__file__).parent.parent.parent
        dir = str(proj_root / "multi_dim_literature_visualizations")
        visualizer,arxiv_trees = visualize_dags(roots, dags, output_dir=dir,topic=args.topic)

        ## 为生成related work，先生成大纲
        print("正在生成related work大纲...")
        outline_prompt = generate_related_work_outline_prompt_various_1(target_abstract,arxiv_trees)
        outline = self.llm_client.get_completion(outline_prompt)
        outline = json_repair.loads(outline)
        

        print("outline:")
        print(outline)
        
        subsection_titles = outline["outline"]
            
        
        args.dimensions = subsection_titles
        roots,dags,id2node,label2node = build_dags(args)
        results = label_papers_by_topic(
            args, 
            internal_collection,
            subsection_titles
        )
        print(results)
        
        update_roots_with_labels(roots, results, internal_collection, args)
        grouped = {dim:[
        {
            "paper_id":pid,
            "title":paper.title,
            "abstract":paper.abstract,
            "summary":paper.summary,
            "citations":paper.citations
        }
        for pid,paper in roots[dim].papers.items()
    ] for dim in args.dimensions}
        
        dim_1 = args.dimensions[0]
        dim_2 = args.dimensions[1]
        grouped_dim_1 = grouped[dim_1]
        grouped_dim_2 = grouped[dim_2]
        
        
        end_time_DAG = time.time()
        
        start_time_related_work = time.time()
        # 生成related work
        prompt = generate_related_work_prompt_with_arxiv_trees(target_abstract,args.dimensions,grouped)
        with open("prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        related_work_with_citations = self.llm_client.get_completion(prompt)
        related_work_with_citations = related_work_with_citations.replace("```json","").replace("```","")
        related_work_with_citations = json_repair.loads(related_work_with_citations)
        print("related_work_with_citations:")
        print(related_work_with_citations)
        related_work = related_work_with_citations["related_work"]
        citations = related_work_with_citations["cite_ids"]
        print("related_work:")
        print(related_work)
        print("citations:")
        print(citations)
        end_time_related_work = time.time()
        
        start_time_extract_cited_sentences = time.time()
        # prompt_for_extract_cited_sentences = process_data_for_extract_cited_sentences(related_work)
        # cited_sentences = self.llm_client.get_completion(prompt_for_extract_cited_sentences)
        # cited_sentences = json_repair.loads(cited_sentences)
        # print("cited_sentences:")
        # print(cited_sentences)
        
        # judge_gemini = Judge(model="google/gemini-2.5-flash")
        # judge_deepseek = Judge(model="deepseek-chat")
        
        # ids = [i for i in citations if "paper_id" in i and i["paper_id"] is not None]
        
        # selected_papers = relevant_pages
        messages = [
            {
                "role":"system",
                "content":"You are a helpful literature review assistant that can do comparison, and accurate attribution."
            },
            {
                "role":"user",
                "content":prompt
            },
            {
                "role":"assistant",
                "content":"related_work to be revised: " + related_work + "\n" + "citations: " + str(citations)
            }
        ]
        
        client = DeepSeekClientMultiTurn(
            api_key = os.environ.get("DEEPSEEK_API_KEY", ""),
            model_name = "deepseek-chat"
        )
        start_time_feedback = time.time()
        feedback_prompt = generate_original_related_work_feedback_prompt(related_work)
        
        feedback = self.llm_client.get_completion(feedback_prompt)
        print("feedback:")
        print(feedback)
        
        prompt_for_revision = generate_related_work_revision_prompt(target_abstract,related_work,feedback,citations,args.dimensions)
        messages.append({
            "role":"user",
            "content":prompt_for_revision
        })
        related_work_revision = client.get_completion(messages)
        print("related_work_revision:")
        print(related_work_revision)
        related_work_revision = related_work_revision.replace("```json","").replace("```","")
        related_work_revision_dict = json_repair.loads(related_work_revision)
        related_work_revision = related_work_revision_dict["related_work"]
        citations = related_work_revision_dict["cite_ids"]
        end_time_feedback = time.time()
        
        related_work_revision,quotes_with_citation_info,validation_results,error_types = self.validate_and_correct_citations(related_work_revision,selected_papers,citations,arxivid)
        end_time = time.time()
        filtered_citations: list[str] = filter_citations(
            related_works_section=related_work, citation_strings=[obj["citations"] if "citations" in obj else "" for obj in selected_papers]
        )
        
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        results_dir = PROJECT_ROOT / f"results/{arxivid.replace('.','_')}"
        # 保存
        results_dir.mkdir(exist_ok=True,parents=True)
        results_dict = {
            "related_work": related_work_revision,
            "citations": citations,
            "quotes_with_citation_info": quotes_with_citation_info,
            "validation_results": validation_results,
            "error_types": error_types
        }
        with open(results_dir / f"results_{breadth}_{depth}_{diversity}.json", "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)
        print("filtered_citations:")
        print(filtered_citations)
        if status_callback:
            status_callback(8, f"Generated related work section with {len(filtered_citations)} citations")
        args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods"]
        
        
        dag_time = end_time_DAG - start_time_DAG
        related_work_time = end_time_related_work - start_time_related_work
        feedback_time = end_time_feedback - start_time_feedback
        all_time = end_time - start_time
        print("Time taken for DAG:", round(dag_time, 2), "seconds")
        print("Time taken for related work:", round(related_work_time, 2), "seconds")
        print("Time taken for feedback:", round(feedback_time, 2), "seconds")
        print("Time taken for all process:", round(all_time, 2), "seconds")
        time_dict = {
            "dag_time": dag_time,
            "related_work_time": related_work_time,
            "feedback_time": feedback_time,
            "all_time": all_time
        }
        prompt = generate_related_work_revise_citations_format(related_work_revision)
        related_work_revision = self.llm_client.get_completion(prompt)
        related_work_revision = related_work_revision.replace("```json","").replace("```","")
        print("related_work_revision:")
        print(related_work_revision)
        final= {"related_works": related_work_revision, "citations": filtered_citations, "selected_papers": selected_papers,"time_dict": time_dict}
            
        args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods"]
        
        
        return final
    
    def generate_related_work_MACG_Only_Retrieval(
            self,
            abstract: str,
            breadth: int,
            depth: int,
            diversity: float,
            arxivid: str,
            status_callback: Optional[Callable] = None,
    ):
        if status_callback:
            status_callback(1, "Initializing")
        print("摘要：", abstract)
        print("向量化摘要并在milvus库中检索...")
        
        
        # 向量化摘要并在milvus库中检索  
        embedded_abstract = self.sentence_embedding_model.encode(abstract)
        topic = self.topic_model.transform(abstract)
        topic_id = topic[0][0]
        
        # 检索milvus库中与摘要最相似的论文
        if status_callback:
            status_callback(2, "Querying Vector DB for matches (this may take a while)")
            
        query_data: list[list[dict]] = self.db_client.search(
            collection_name = "abstracts",
            data = [embedded_abstract],
            limit = 6 * breadth,
            anns_field = "embedding",
            search_params = {"metric_type": "COSINE", "params": {}},
            output_fields = ["embedding"],
        )
        
        if status_callback:
            status_callback(3, f"Retrieved {len(query_data[0])} papers from the DB")
            
        # 清理DB响应数据
        papers_data: list[dict] = query_data[0]
        for obj in papers_data:
            obj["embedding"] = obj["entity"]["embedding"]
            obj.pop("entity")
            
        # 选择一个长列表的论文
        selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
            paper_data = papers_data,
            k = 3 * breadth,
            diversity_weight = diversity)
        
        if status_callback:
            status_callback(4, f"Selected {len(selected_papers)} papers for the longlist, retrieving full text(s) (this might take a while)")
            
        # 生成每个论文的页面嵌入
        page_embeddings: list[list[dict]] = []
        for paper in selected_papers:
            arxiv_id = paper["id"]
            result = process_arxiv_paper_with_embeddings(arxiv_id, self.topic_model)
            if result:
                page_embeddings.append(result)
                
        if status_callback:
            status_callback(5, f"Generated page embeddings for {len(page_embeddings)} papers")
            
        # 生成一个短列表的论文（最多k页每篇，最多b篇）
        relevant_pages: list[dict] = select_diverse_pages_for_top_b_papers(
            paper_embeddings=page_embeddings,
            input_string=abstract,
            topic_model=self.topic_model,
            k=depth,
            b=breadth,
            diversity_weight=diversity,
            skip_first=False,
        )
        start_time = time.time()
        # with open("relevant_pages.json", "w", encoding="utf-8") as f:
        #     json.dump(relevant_pages, f, indent=4, ensure_ascii=False)

        if status_callback:
            status_callback(6, f"Selected {len(relevant_pages)} papers for the shortlist")
        
        selected_papers = []
        

        internal_collection = {}
        data = []
        id = 0
        start_time_summarization = time.time()
        # Generate summaries for individual papers (taking all relevant pages into account)
        for obj in relevant_pages[:]:
            # Because paper_id != arXiv_id -> retrieve arXiv id/
            arxiv_id = papers_data[obj["paper_id"]]["id"]
            arxiv_abstract = get_arxiv_abstract(arxiv_id)
            text_segments = obj["text"]
            obj["cite_ids"] = [id]
            title = get_arxiv_title(arxiv_id)
            
            # Create prompt
            prompt = generate_summary_prompt_with_page_content(
                abstract_source_paper=abstract,
                abstract_to_be_cited=arxiv_abstract,
                page_text_to_be_cited=text_segments,
                sentence_count=5,
            )
            
            internal_collection[id] = Paper(
                id, 
                title, 
                arxiv_abstract, 
                label_opts=["tasks", "datasets", "methodologies", "evaluation_methods"], 
                internal=True
            )
            temp_dict = {"Title": title, "Abstract": arxiv_abstract}
            data.append(temp_dict)            
            # Use the appropriate LLM client based on the provider
            response: str = self.llm_client.get_completion(prompt)
            obj["summary"] = response
            
            obj["citation"] = get_arxiv_citation(arxiv_id)
            internal_collection[id].summary = response
            internal_collection[id].citations = get_arxiv_citation(arxiv_id)
            idx = {
                "cite_ids":[id],
                "title":title,
                "abstract":arxiv_abstract,
                "summary":response,
                "arxiv_id":arxiv_id,
                "citations":get_arxiv_citation(arxiv_id)
            }
            selected_papers.append(idx)
            id += 1
        # with open("internal_collection.json", "w", encoding="utf-8") as f:
        #     json.dump(data, f, indent=4, ensure_ascii=False)
        end_time_summarization = time.time()
        
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        time = datetime.date.today()
        results_dir = PROJECT_ROOT / f"selected_papers/{arxivid.replace('.','_')}/{time}"
        results_dir.mkdir(exist_ok=True,parents=True)
        with open(f"{results_dir}/{breadth}_{depth}_{diversity}.json", "w", encoding="utf-8") as f:
            json.dump(selected_papers, f, indent=4, ensure_ascii=False)
        
        
        
        
    
    def generate_related_work_MACG(
            self,
            abstract: str,
            breadth: int,
            depth: int,
            diversity: float,
            arxivid: str,
            status_callback: Optional[Callable] = None,
    ):
        token_usage = {
        "stages": {
            "summarization": 0,
            "topic": 0,
            "outline": 0,
            "DAG": 0,
            "related_work": 0,
            "feedback": 0,
            "revision": 0,
            "validation": 0
        },
        "total_input_tokens": 0,
        "total_output_tokens": 0
    }
        if status_callback:
            status_callback(1, "Initializing")
        print("摘要：", abstract)
        print("向量化摘要并在milvus库中检索...")
        
        
        # 向量化摘要并在milvus库中检索  
        embedded_abstract = self.sentence_embedding_model.encode(abstract)
        topic = self.topic_model.transform(abstract)
        topic_id = topic[0][0]
        
        
        # 检索milvus库中与摘要最相似的论文
        if status_callback:
            status_callback(2, "Querying Vector DB for matches (this may take a while)")
            
        query_data: list[list[dict]] = self.db_client.search(
            collection_name = "abstracts",
            data = [embedded_abstract],
            limit = 6 * breadth,
            anns_field = "embedding",
            search_params = {"metric_type": "COSINE", "params": {}},
            output_fields = ["embedding"],
        )
        
        if status_callback:
            status_callback(3, f"Retrieved {len(query_data[0])} papers from the DB")
            
        # 清理DB响应数据
        papers_data: list[dict] = query_data[0]
        for obj in papers_data:
            obj["embedding"] = obj["entity"]["embedding"]
            obj.pop("entity")
            
        # 选择一个长列表的论文
        selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
            paper_data = papers_data,
            k = 3 * breadth,
            diversity_weight = diversity)
        
        if status_callback:
            status_callback(4, f"Selected {len(selected_papers)} papers for the longlist, retrieving full text(s) (this might take a while)")
            
        # 生成每个论文的页面嵌入
        page_embeddings: list[list[dict]] = []
        for paper in selected_papers:
            arxiv_id = paper["id"]
            result = process_arxiv_paper_with_embeddings(arxiv_id, self.topic_model)
            if result:
                page_embeddings.append(result)
                
        if status_callback:
            status_callback(5, f"Generated page embeddings for {len(page_embeddings)} papers")
            
        # 生成一个短列表的论文（最多k页每篇，最多b篇）
        relevant_pages: list[dict] = select_diverse_pages_for_top_b_papers(
            paper_embeddings=page_embeddings,
            input_string=abstract,
            topic_model=self.topic_model,
            k=depth,
            b=breadth,
            diversity_weight=diversity,
            skip_first=False,
        )
        start_time = time.time()
        # with open("relevant_pages.json", "w", encoding="utf-8") as f:
        #     json.dump(relevant_pages, f, indent=4, ensure_ascii=False)

        if status_callback:
            status_callback(6, f"Selected {len(relevant_pages)} papers for the shortlist")
        selected_papers_for_id = selected_papers
        selected_papers = []
        internal_collection = {}
        data = []
        id = 0
        start_time_summarization = time.time()
        
        # 创建summarization prompts目录
        summarization_prompts_dir = results_dir / "prompts" / "00_summarization"
        summarization_prompts_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate summaries for individual papers (taking all relevant pages into account)
        for obj in relevant_pages[1:]:
            # Because paper_id != arXiv_id -> retrieve arXiv id/
            # arxiv_id = papers_data[obj["paper_id"]]["id"]
            # arxiv_abstract = get_arxiv_abstract(arxiv_id)
            # text_segments = obj["text"]
            # obj["cite_ids"] = [id]
            # title = get_arxiv_title(arxiv_id)
            
            arxiv_id = selected_papers_for_id[obj["paper_id"]]["id"]
            arxiv_abstract = get_arxiv_abstract(arxiv_id)
            title = get_arxiv_title(arxiv_id)
            text_segments = obj["text"]
            
            # Create prompt
            prompt = generate_summary_prompt_with_page_content(
                abstract_source_paper=abstract,
                abstract_to_be_cited=arxiv_abstract,
                page_text_to_be_cited=text_segments,
                sentence_count=5,
            )
            
            token_usage["stages"]["summarization"] += self._count_tokens(prompt)
            token_usage["total_input_tokens"] += self._count_tokens(prompt)
            
            internal_collection[id] = Paper(
                id, 
                title, 
                arxiv_abstract, 
                label_opts=["tasks", "datasets", "methodologies", "evaluation_methods"], 
                internal=True
            )
            temp_dict = {"Title": title, "Abstract": arxiv_abstract}
            
            data.append(temp_dict)            
            # Use the appropriate LLM client based on the provider
            response: str = self.llm_client.get_completion(prompt)
            
            token_usage["stages"]["summarization"] += self._count_tokens(response)
            token_usage["total_output_tokens"] += self._count_tokens(response)
            obj["summary"] = response
            
            obj["citation"] = get_arxiv_citation(arxiv_id)
            
            internal_collection[id].summary = response
            internal_collection[id].citations = get_arxiv_citation(arxiv_id)
            idx = {
                "cite_ids":[id],
                "title":title,
                "abstract":arxiv_abstract,
                "summary":response,
                "arxiv_id":arxiv_id,
                "citations":get_arxiv_citation(arxiv_id)
            }
            selected_papers.append(idx)
            
            # 保存summarization prompt和response
            with open(summarization_prompts_dir / f"paper_{id:02d}_{arxiv_id}_prompt.txt", "w", encoding="utf-8") as f:
                f.write("="*80 + "\n")
                f.write(f"PAPER SUMMARIZATION PROMPT - {title}\n")
                f.write(f"Paper ID: {id} | ArXiv ID: {arxiv_id}\n")
                f.write("="*80 + "\n\n")
                f.write(prompt)
            
            with open(summarization_prompts_dir / f"paper_{id:02d}_{arxiv_id}_response.txt", "w", encoding="utf-8") as f:
                f.write("="*80 + "\n")
                f.write(f"PAPER SUMMARIZATION RESPONSE - {title}\n")
                f.write(f"Paper ID: {id} | ArXiv ID: {arxiv_id}\n")
                f.write("="*80 + "\n\n")
                f.write(response)
            
            id += 1
        # with open("internal_collection.json", "w", encoding="utf-8") as f:
        #     json.dump(data, f, indent=4, ensure_ascii=False)
        end_time_summarization = time.time()
        
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        results_dir = PROJECT_ROOT / f"selected_papers/{arxivid.replace('.','_')}"
        

        # 确保目录存在
        results_dir.mkdir(exist_ok=True,parents=True)
        with open(f"{results_dir}/{breadth}_{depth}_{diversity}.json", "w", encoding="utf-8") as f:
            json.dump(selected_papers, f, indent=4, ensure_ascii=False)
        
        # with open("/home/liujian/internal_collection.json", "r", encoding="utf-8") as f:
        #       internal_collection = json.load(f)
        if status_callback:
            status_callback(7, "Generated summaries of papers (and their pages)")
        
        start_time_DAG = time.time()
        
        # 创建prompts输出目录
        prompts_output_dir = results_dir / "prompts"
        prompts_output_dir.mkdir(exist_ok=True, parents=True)
        
        # 生成本篇文章的topic
        topic_prompt = generate_brief_topic_prompt(abstract)
        # 保存topic prompt
        with open(prompts_output_dir / "01_topic_prompt.txt", "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("TOPIC GENERATION PROMPT\n")
            f.write("="*80 + "\n\n")
            f.write(topic_prompt)
        
        topic = self.llm_client.get_completion(topic_prompt)
        
        # 保存topic response
        with open(prompts_output_dir / "01_topic_response.txt", "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("TOPIC GENERATION RESPONSE\n")
            f.write("="*80 + "\n\n")
            f.write(topic)
        
        print("topic:")
        print(topic)
        token_usage["stages"]["topic"] += self._count_tokens(topic_prompt)
        token_usage["total_input_tokens"] += self._count_tokens(topic_prompt)
        # 生成文献树
        args.topic = topic
        
        roots,dags = run_dag_to_classifier(
            args,
            internal_collection
        )

        print("正在生成可视化...")
        
        proj_root = Path(__file__).parent.parent.parent
        dir = str(proj_root / "multi_dim_literature_visualizations")
        visualizer,arxiv_trees = visualize_dags(roots, dags, output_dir=dir,topic=args.topic)

        ## 为生成related work，先生成大纲
        print("正在生成related work大纲...")
        outline_prompt = generate_related_work_outline_prompt_various_1(abstract,arxiv_trees)
        
        # 保存outline prompt
        with open(prompts_output_dir / "02_outline_prompt.txt", "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("OUTLINE GENERATION PROMPT\n")
            f.write("="*80 + "\n\n")
            f.write(outline_prompt)
        
        token_usage["stages"]["DAG"] += self._count_tokens(outline_prompt)
        token_usage["total_input_tokens"] += self._count_tokens(outline_prompt)
        outline = self.llm_client.get_completion(outline_prompt)
        outline = json_repair.loads(outline)
        
        # 保存outline response
        with open(prompts_output_dir / "02_outline_response.txt", "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("OUTLINE GENERATION RESPONSE\n")
            f.write("="*80 + "\n\n")
            f.write(json.dumps(outline, ensure_ascii=False, indent=4))

        print("outline:")
        print(outline)
        
        subsection_titles = outline["outline"]
        token_usage["stages"]["DAG"] += self._count_tokens(outline_prompt)
        token_usage["total_input_tokens"] += self._count_tokens(outline_prompt)
        
        token_usage["stages"]["outline"] += self._count_tokens(str(outline))
        token_usage["total_output_tokens"] += self._count_tokens(str(outline))
        args.dimensions = subsection_titles
        roots,dags,id2node,label2node = build_dags(args)

        results = label_papers_by_topic(
            args, 
            internal_collection,
            subsection_titles
        )
        print(results)

        update_roots_with_labels(roots, results, internal_collection, args)
        grouped = {dim:[
        {
            "paper_id":pid,
            "title":paper.title,
            "abstract":paper.abstract,
            "summary":paper.summary,
            "citations":paper.citations
        }
        for pid,paper in roots[dim].papers.items()
    ] for dim in args.dimensions}
        
        dim_1 = args.dimensions[0]
        dim_2 = args.dimensions[1]
        grouped_dim_1 = grouped[dim_1]
        grouped_dim_2 = grouped[dim_2]
        
        
        end_time_DAG = time.time()
        
        start_time_related_work = time.time()
        # 生成related work
        prompt = generate_related_work_prompt_with_arxiv_trees(abstract,args.dimensions,grouped)
        
        # 保存related work generation prompt
        with open(prompts_output_dir / "03_related_work_generation_prompt.txt", "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("RELATED WORK GENERATION PROMPT\n")
            f.write("="*80 + "\n\n")
            f.write(prompt)
        
        token_usage["stages"]["related_work"] += self._count_tokens(prompt)
        token_usage["total_input_tokens"] += self._count_tokens(prompt)
        related_work_with_citations = self.llm_client.get_completion(prompt)
        related_work_with_citations = related_work_with_citations.replace("```json","").replace("```","")
        related_work_with_citations = json_repair.loads(related_work_with_citations)
        
        # 保存related work generation response
        with open(prompts_output_dir / "03_related_work_generation_response.txt", "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("RELATED WORK GENERATION RESPONSE\n")
            f.write("="*80 + "\n\n")
            f.write(json.dumps(related_work_with_citations, ensure_ascii=False, indent=4))
        
        print("related_work_with_citations:")
        print(related_work_with_citations)
        token_usage["stages"]["related_work"] += self._count_tokens(str(related_work_with_citations))
        token_usage["total_output_tokens"] += self._count_tokens(str(related_work_with_citations))
        related_work = related_work_with_citations["related_work"]
        citations = related_work_with_citations["cite_ids"]
        print("related_work:")
        print(related_work)
        print("citations:")
        print(citations)
        
        end_time_related_work = time.time()
        
        messages = [
            {
                "role":"system",
                "content":"You are a helpful literature review assistant that can do comparison, and accurate attribution."
            },
            {
                "role":"user",
                "content":prompt
            },
            {
                "role":"assistant",
                "content":"related_work to be revised: " + related_work + "\n" + "citations: " + str(citations)
            }
        ]
        
        client = DeepSeekClientMultiTurn(
            api_key = os.environ.get("DEEPSEEK_API_KEY", ""),
            model_name = "deepseek-chat"
        )
        start_time_feedback = time.time()
        feedback_prompt = generate_original_related_work_feedback_prompt(related_work)
        
        # 保存feedback prompt
        with open(prompts_output_dir / "04_feedback_prompt.txt", "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("FEEDBACK GENERATION PROMPT\n")
            f.write("="*80 + "\n\n")
            f.write(feedback_prompt)
        
        token_usage["stages"]["feedback"] += self._count_tokens(feedback_prompt)
        token_usage["total_input_tokens"] += self._count_tokens(feedback_prompt)
        feedback = self.llm_client.get_completion(feedback_prompt)
        
        # 保存feedback response
        with open(prompts_output_dir / "04_feedback_response.txt", "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("FEEDBACK GENERATION RESPONSE\n")
            f.write("="*80 + "\n\n")
            f.write(feedback)
        
        print("feedback:")
        print(feedback)
        token_usage["stages"]["feedback"] += self._count_tokens(str(feedback))
        token_usage["total_output_tokens"] += self._count_tokens(feedback)
        prompt_for_revision = generate_related_work_revision_prompt(abstract,related_work,feedback,citations,args.dimensions)
        
        # 保存revision prompt
        with open(prompts_output_dir / "05_revision_prompt.txt", "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("REVISION GENERATION PROMPT\n")
            f.write("="*80 + "\n\n")
            f.write(prompt_for_revision)
        
        messages.append({
            "role":"user",
            "content":prompt_for_revision
        })
        token_usage["stages"]["revision"] += self._count_tokens(prompt_for_revision)
        token_usage["total_input_tokens"] += self._count_tokens(prompt_for_revision)
        related_work_revision = client.get_completion(messages)
        
        print("related_work_revision:")
        print(related_work_revision)
        related_work_revision = related_work_revision.replace("```json","").replace("```","")
        token_usage["stages"]["revision"] += self._count_tokens(related_work_revision)
        token_usage["total_output_tokens"] += self._count_tokens(str(related_work_revision))
        related_work_revision_dict = json_repair.loads(related_work_revision)
        
        # 保存revision response
        with open(prompts_output_dir / "05_revision_response.txt", "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("REVISION GENERATION RESPONSE\n")
            f.write("="*80 + "\n\n")
            f.write(json.dumps(related_work_revision_dict, ensure_ascii=False, indent=4))
        
        related_work_revision = related_work_revision_dict["related_work"]
        citations = related_work_revision_dict["cite_ids"]
        end_time_feedback = time.time()
          
        related_work_revision,quotes_with_citation_info,validation_results,error_types = self.validate_and_correct_citations(related_work_revision,selected_papers,citations,arxivid)
        token_usage["stages"]["validation"] += self._count_tokens(prompt_for_revision)
        token_usage["total_input_tokens"] += self._count_tokens(prompt_for_revision)
        end_time = time.time()
        filtered_citations: list[str] = filter_citations(
            related_works_section=related_work, citation_strings=[obj["citation"] for obj in relevant_pages[1:]]
        )

        PROJECT_ROOT = Path(__file__).parent.parent.parent
        results_dir = PROJECT_ROOT / f"results/{arxivid.replace('.','_')}"
        # 保存
        results_dir.mkdir(exist_ok=True,parents=True)
        results_dict = {
            "related_work": related_work_revision,
            "quotes_with_citation_info": quotes_with_citation_info,
            "citations": citations,
            "validation_results": validation_results,
            "error_types": error_types
        }
        with open(results_dir / f"results_{breadth}_{depth}_{diversity}.json", "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)
        print("filtered_citations:")
        print(filtered_citations)
        date = datetime.date.today()
        # os.makedirs(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}", exist_ok=True)
        # with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/related_work_with_citations.txt", "w", encoding="utf-8") as f:
        #     f.write("="*50 + "\n" + "related_work" + "\n" + "="*50 + "\n" + related_work + "\n" + "="*50 + "\n" + "citations" + "\n" + "="*50 + "\n" + str(filtered_citations))
        # with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/grouped_dim_1.json", "w", encoding="utf-8") as f:
        #     json.dump(grouped_dim_1,f,ensure_ascii=False,indent=4)
        # with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/grouped_dim_2.json", "w", encoding="utf-8") as f:
        #     json.dump(grouped_dim_2,f,ensure_ascii=False,indent=4)
        # with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/related_work_revision.txt", "w", encoding="utf-8") as f:
        #     f.write("="*50 + "\n" + "related_work_revision" + "\n" + "="*50 + "\n" + related_work_revision + "\n" + "="*50 + "\n" + "citations" + "\n" + "="*50 + "\n" + str(filtered_citations))
        if status_callback:
            status_callback(8, f"Generated related work section with {len(filtered_citations)} citations")
        args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods"]
        
        
        dag_time = end_time_DAG - start_time_DAG
        related_work_time = end_time_related_work - start_time_related_work
        feedback_time = end_time_feedback - start_time_feedback
        summarization_time = end_time_summarization - start_time_summarization
        all_time = end_time - start_time
        print("Time taken for summarization:", round(summarization_time, 2), "seconds")
        print("Time taken for DAG:", round(dag_time, 2), "seconds")
        print("Time taken for related work:", round(related_work_time, 2), "seconds")
        print("Time taken for feedback:", round(feedback_time, 2), "seconds")
        print("Time taken for all process:", round(all_time, 2), "seconds")
        time_dict = {
            "summarization_time": summarization_time,
            "dag_time": dag_time,
            "related_work_time": related_work_time,
            "feedback_time": feedback_time,
            "all_time": all_time
        } 
        
        final= {"related_works": related_work_revision, "citations": filtered_citations, "selected_papers": relevant_pages,"time_dict": time_dict}
        # with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/final.json", "w", encoding="utf-8") as f:
        #     json.dump(final,f,ensure_ascii=False,indent=4)
            
        args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods"]
        
        token_usage["stages"]["validation"] += self._count_tokens(str(quotes_with_citation_info))
        token_usage["total_input_tokens"] += self._count_tokens(str(quotes_with_citation_info)) 
        
        token_usage["stages"]["validation"] += self._count_tokens(str(validation_results))
        token_usage["total_output_tokens"] += self._count_tokens(str(validation_results))
        
        
        return final,token_usage
    
    def validate_and_correct_citations(self, related_work, selected_papers, citations,arxiv_id):
        """
        验证和纠正引用错误的内联方法
        """
        # 1. 提取引用的句子
        prompt_for_extract_cited_sentences = process_data_for_extract_cited_sentences(related_work)
        cited_sentences = self.llm_client.get_completion(prompt_for_extract_cited_sentences)
        cited_sentences = json_repair.loads(cited_sentences)
        
        # 2. 构建引用验证数据
        # The code snippet you provided is a comment in Python. It appears to be describing a variable
        # or data structure called `quotes_with_citation_info`. The triple hash symbols (`
        quotes_with_citation_info = self._build_citation_verification_data(
            cited_sentences, selected_papers, citations
        )
        
        # 3. 使用双模型验证
        validation_results = self._validate_citations_with_dual_models(quotes_with_citation_info,related_work,citations,selected_papers)
        
        # 4. 分类错误类型
        error_types = self._classify_citation_errors(validation_results)
        
        # 5. 纠正错误
        corrected_related_work = self._correct_citation_errors(related_work, error_types)
        prompt_for_extract_cited_sentences = process_data_for_extract_cited_sentences(corrected_related_work)
        cited_sentences = self.llm_client.get_completion(prompt_for_extract_cited_sentences)
        cited_sentences = json_repair.loads(cited_sentences)
        quotes_with_citation_info = self._build_citation_verification_data(
            cited_sentences, selected_papers, citations
        )
        
        validation_results = self._validate_citations_with_dual_models(quotes_with_citation_info,related_work,citations,selected_papers)
        
        print("="*50)
        print("claim_precision:")
        print(validation_results["claim_precision"])
        print("citation_precision:")
        print(validation_results["citation_precision"])
        print("reference_precision:")
        print(validation_results["reference_precision"])
        print("citation_density:")
        print(validation_results["citation_density"])
        print("avg_citation_per_sentence:")
        print(validation_results["avg_citation_per_sentence"])
        print("="*50)
        
        
        return corrected_related_work, quotes_with_citation_info, validation_results, error_types
    
    
    def _count_sentences(self,text):
        sentences = re.split(r"[.!?\n]+(?:\s|\n|$)", text.strip())
        sentences = [s for s in sentences if s]
        return sentences
    
    
    def _build_citation_verification_data(
        self, 
        quotes: list[str], 
        selected_papers: list[dict], 
        cite_ids: list[dict]
    ):
        """构建引用验证所需的数据结构"""
        quotes_with_citation_info = {quote: [] for quote in quotes}
        ids = [i for i in cite_ids if "paper_id" in i and i["paper_id"] is not None]
        
        for cited_id in ids:
            for id,selected_paper in enumerate(selected_papers):
                cited_paper_id = cited_id["paper_id"]
                cited_paper_id = cited_paper_id.replace("paper_","")
                if cited_paper_id == "null":
                    continue
                if "cite_ids" not in selected_paper:
                    continue
                try:
                    if int(cited_paper_id) == int(selected_paper["cite_ids"][0]):
                        cited_id["summary"] = selected_paper["summary"]
                        cited_id["abstract"] = selected_paper["abstract"]
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
                if c_text in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["abstract"])
                elif c_text.split(".")[0] in quote and year in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["abstract"])
                elif c_text.split("(")[0] in quote and year in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["abstract"])
                elif c_text.split(",")[0] in quote and year in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["abstract"])
                elif c_text.split("et")[0] in quote and year in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["abstract"])
                elif c_text.split("&")[0] in quote and year in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["abstract"])
                elif c_text.split("and")[0] in quote and year in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["abstract"])
        print("quotes_with_citation_info:")
        print(quotes_with_citation_info)
        return quotes_with_citation_info
    
    def _validate_citations_with_dual_models(self, quotes_with_citation_info,related_work,citations,selected_papers):
        """使用双模型验证引用准确性"""

        yes_gemini = []
        no_gemini = []
        yes_deepseek = []
        no_deepseek = []
        validation_results = {
                "yes_gemini": [],
                "no_gemini": [],
                "yes_deepseek": [],
                "no_deepseek": [],
                "yes_ids": [],
                "no_ids": [],
                "claim_precision": 0,
                "citation_precision": 0,
                "reference_precision": 0,
                "citation_density": 0,
                "avg_citation_per_sentence": 0
            }
        for i,quote in enumerate(quotes_with_citation_info.keys()):
            if len(quotes_with_citation_info[quote]) == 0:
                continue
            q = list(set(quotes_with_citation_info[quote]))
            source = "\n".join(q)
            print(f"source: {source}")
            print(f"quote: {quote}")
            score = judge_gemini._get_pair_score_new(source, quote)
            print(f"{i}. {score} by gemini")
            if score.lower() == "yes":
                yes_gemini.append({"id": i, "claim": quote, "source": source})
            else:
                no_gemini.append({"id": i, "claim": quote, "source": source})
            score = judge_deepseek._get_pair_score_new(source, quote)
            print(f"{i}. {score} by deepseek")
            if score.lower() == "yes":
                yes_deepseek.append({"id": i, "claim": quote, "source": source})
            else:
                no_deepseek.append({"id": i, "claim": quote, "source": source})
            quotes_with_citation_info[quote] = q
        # 计算yes_ids和no_ids,两个模型都认为才算对
        yes_gemini_ids = [(j["id"],j["source"]) for j in yes_gemini]
        yes_deepseek_ids = [(j["id"],j["source"]) for j in yes_deepseek]
        no_gemini_ids = [(j["id"],j["source"]) for j in no_gemini]
        no_deepseek_ids = [(j["id"],j["source"]) for j in no_deepseek]
        yes_ids = list(set(yes_gemini_ids) & set(yes_deepseek_ids))
        # 两个模型一个认为对，一个认为错，则算错
        no_ids = list(set(no_gemini_ids) | set(no_deepseek_ids))
        # 带引用的claim数量
        total_claims = len(quotes_with_citation_info.keys())
        # related work中总共的句子数量
        total_sentences = len(self._count_sentences(related_work))
        
        ## 1.claim_precision 正确引用的claim数量 / 所有claim数量（句子层面的）
        claim_precision = len(yes_ids) / total_claims
        claim_precision = round(claim_precision,3)
        
        ## 2.citation_precision 正确引用数量 / 所有引用数量 （引用层面的）
        correct_source = 0
        for yes_id in yes_ids:
            source = yes_id[1]
            source = source.split("\n")
            correct_source += len(source)
        
        total_citations = len(citations)
        
        citation_precision = correct_source / total_citations
        citation_precision = round(citation_precision,3)
        
        ## 3.reference_precision 被正确引用的不同论文篇数（其实就是正确的引用去重之后） / 参考文献总篇数 （信源层面的）
        correct_reference_source = set()
        for yes_id in yes_ids:
            source = yes_id[1]
            source = source.split("\n")
            for s in source:
                correct_reference_source.add(s)

        ## 全部被引用上的文章数量，将被引用的信源进行去重，然后计算数量
        unique_citations = set()
        for quote in quotes_with_citation_info.keys():
            for citation in quotes_with_citation_info[quote]:
                unique_citations.add(citation)
        reference_precision = len(correct_reference_source) / len(selected_papers) # unique_citations
        reference_precision = round(reference_precision,3)
        
        ## 4. citation_density 引用总数 ÷ 正文句子总数(引用密度层面的)
        citation_density = total_citations / total_sentences
        citation_density = round(citation_density,3)
        
        ## 5. avg_citation_per_sentence 引用总数 ÷ claim总数(引用密度层面的)
        avg_citation_per_sentence = total_citations / total_claims
        avg_citation_per_sentence = round(avg_citation_per_sentence,3)
            
        validation_results = {
            "yes_gemini": yes_gemini,
            "no_gemini": no_gemini,
            "yes_deepseek": yes_deepseek,
            "no_deepseek": no_deepseek,
            "yes_ids": yes_ids,
            "no_ids": no_ids,
            "claim_precision": claim_precision,
            "citation_precision": citation_precision,
            "reference_precision": reference_precision,
            "citation_density": citation_density,
            "avg_citation_per_sentence": avg_citation_per_sentence
        }  
            
            
        return validation_results

    def _classify_citation_errors(self, validation_results):
        """分类引用错误类型"""
        yes_ids = validation_results["yes_ids"]
        no_ids = validation_results["no_ids"]
        no_ids = [j[0] for j in no_ids]
        yes_gemini = validation_results["yes_gemini"]
        no_gemini = validation_results["no_gemini"]
        yes_deepseek = validation_results["yes_deepseek"]
        no_deepseek = validation_results["no_deepseek"]
        
        direct_contradiction = 0
        information_not_present = 0
        misrepresentation = 0
        incorrect_attribution = 0
        other = 0
        
        error_types = []
        for quote_id in no_ids:
            gemini_item = next((item for item in no_gemini if item["id"] == quote_id), None)
            deepseek_item = next((item for item in no_deepseek if item["id"] == quote_id), None)
            
            if gemini_item:
                claim = gemini_item["claim"]
                source = gemini_item["source"]
            elif deepseek_item:
                claim = deepseek_item["claim"]
                source = deepseek_item["source"]
            else:
                print(f"Warning: quote_id {quote_id} not found in either no_gemini or no_deepseek")
                continue
            
            try:
                prompt = process_data_for_classify_errors(claim,source)
                result = self.llm_client.get_completion(prompt)
                result = result.replace("```json", "").replace("```", "")
                result = json_repair.loads(result)
                print(result)
                print(result["error_type"])
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
                if isinstance(result, dict):
                    result["id"] = quote_id
                    result["claim"] = claim
                    result["source"] = source
                    result["errors_count"] = errors_count
                    error_types.append(result)
                else:
                    print(f"Error: Unexpected result format: {result}")
                    continue
                
            except Exception as e:
                print(e)
                continue
            print("-"*100)
        
        print(f"Direct Contradiction: {direct_contradiction}")
        print(f"Information Not Present / Unsubstantiated: {information_not_present}")
        print(f"Misrepresentation / Imprecise Wording: {misrepresentation}")
        print(f"Incorrect Attribution: {incorrect_attribution}")
        print(f"Other: {other}")
        
        
        
        return error_types
    
    def _correct_citation_errors(self, related_work, error_types):
        """纠正引用错误"""
        correct_count = 0
        Direct_Contradiction = 0
        Information_Not_Present = 0
        Misrepresentation = 0
        Incorrect_Attribution = 0
        Other = 0
        for error in error_types:
            source = error["source"]
            claim = error["claim"]
            error_type = error["error_type"]
            error_description = error["error_description"]
            
            prompt = process_data_for_correct_citation_errors(source,claim,error_type,error_description)
            result = self.llm_client.get_completion(prompt)
            result = result.replace("```json", "").replace("```", "")
            result = json_repair.loads(result)
            error["corrected_claim"] = result["corrected_claim"]
            error["explanation"] = result["explanation"]
            error["key_changes"] = result["key_changes"]
            print(f"corrected_claim: {result['corrected_claim']}")
            print(f"explanation: {result['explanation']}")
            score = judge_gemini._get_pair_score_new(source, result["corrected_claim"])
            print(f"score: {score}")
            print("-"*100)
            if score == "Yes":
                correct_count += 1
            else:
                prompt = process_data_for_classify_errors(result["corrected_claim"],source)
                result = self.llm_client.get_completion(prompt)
                result = result.replace("```json", "").replace("```", "")
                result = json_repair.loads(result)
                error_type = result["error_type"]
                error_description = result["error_description"]
                if "Direct Contradiction" in error_type:
                    Direct_Contradiction += 1
                elif "Information Not Present / Unsubstantiated" in error_type:
                    Information_Not_Present += 1
                elif "Misrepresentation / Imprecise Wording" in error_type:
                    Misrepresentation += 1
                elif "Incorrect Attribution" in error_type:
                    Incorrect_Attribution += 1
                else:
                    Other += 1
        print(f"Direct Contradiction: {Direct_Contradiction}")
        print(f"Information Not Present / Unsubstantiated: {Information_Not_Present}")
        print(f"Misrepresentation / Imprecise Wording: {Misrepresentation}")
        print(f"Incorrect Attribution: {Incorrect_Attribution}")
        print(f"Other: {Other}")
        
        
        related_work_revision = related_work
        for error in error_types:
            error_claim = error["claim"]
            related_work_revision = related_work_revision.replace(error_claim,error["corrected_claim"])
        print(f"related_work_revision: {related_work_revision}")
        return related_work_revision

    
    def generate_related_work(
        self,
        abstract: str,
        breadth: int,
        depth: int,
        diversity: float,
        arxivid: str,
        status_callback: Optional[Callable] = None,
    ) -> dict[str, str | list[str] | list[dict]]:
    
        """
        Generate a related work section based on an abstract.

        Args:
            abstract: The input abstract text
            breadth: Number of papers to consider
            depth: Number of pages to extract from each paper
            diversity: Diversity factor for paper selection (0-1)
            status_callback: Callback function that will update jobs according to the function progress

        Returns:
            Dictionary with 'related_works' text and 'citations' list
        """
        if status_callback:
            status_callback(1, "Initializing")

        embedded_abstract = self.sentence_embedding_model.encode(abstract)
        # topic = self.topic_model.transform(abstract)
        # topic_id = topic[0][0]

        # Query Milvus Vector DB
        if status_callback:
            status_callback(2, "Querying Vector DB for matches (this may take a while)")

        query_data: list[list[dict]] = self.db_client.search(
            collection_name="abstracts",
            data=[embedded_abstract],
            limit=6 * breadth,
            anns_field="embedding",
            # filter = f'topic == {topic_id}',
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["embedding"],
        )

        if status_callback:
            status_callback(3, f"Retrieved {len(query_data[0])} papers from the DB")

        # Clean DB response data
        papers_data: list[dict] = query_data[0]
        for obj in papers_data:
            obj["embedding"] = obj["entity"]["embedding"]
            obj.pop("entity")

        # Select a longlist of papers
        selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
            paper_data=papers_data, k=3 * breadth, diversity_weight=diversity
        )

        if status_callback:
            status_callback(
                4,
                f"Selected {len(selected_papers)} papers for the longlist, retrieving full text(s)"
                f" (this might take a while)",
            )

        # Generate embeddings of each page of every paper in the longlist
        page_embeddings: list[list[dict]] = []
        for paper in selected_papers:
            arxiv_id = paper["id"]
            result = process_arxiv_paper_with_embeddings(arxiv_id, self.topic_model)
            text = result[0]["text"]
            
            dir = Path(__file__).parent.parent.parent / "text" / arxivid
            dir.mkdir(exist_ok = True,parents = True)
            with open(dir / f"text_{arxiv_id}.txt", "w", encoding="utf-8") as f:
                f.write(text)
            
            if result:
                page_embeddings.append(result)

        if status_callback:
            status_callback(5, f"Generated page embeddings for {len(page_embeddings)} papers")

        # Generate shortlist of papers (at most k pages per paper, at most b papers in total)
        relevant_pages: list[dict] = select_diverse_pages_for_top_b_papers(
            paper_embeddings=page_embeddings,
            input_string=abstract,
            topic_model=self.topic_model,
            k=depth,
            b=breadth,
            diversity_weight=diversity,
            skip_first=False,
        )

        if status_callback:
            status_callback(6, f"Selected {len(relevant_pages)} papers for the shortlist")
            
        selected_papers_for_id = selected_papers
        selected_papers = []

        internal_collection = {}
        data = []
        id = 0
            # with open("relevant_pages.json", "w", encoding="utf-8") as f:
            #     json.dump(relevant_pages, f, indent=4, ensure_ascii=False)
        # Generate summaries for individual papers (taking all relevant pages into account)
        for obj in relevant_pages[1:]:
            # Because paper_id != arXiv_id -> retrieve arXiv id/
            # 不要使用papers_data中的索引，要使用selected_papers中的索引，这样才能对齐
            # print(obj["paper_id"])
            # arxiv_id = papers_data[obj["paper_id"]]["id"]
            # print("arxiv_id from obj:")
            # print(arxiv_id)
            # arxiv_abstract = get_arxiv_abstract(arxiv_id)
            # text_segments = obj["text"]
            # title = get_arxiv_title(arxiv_id)
            
            arxiv_id = selected_papers_for_id[obj["paper_id"]]["id"]
            arxiv_abstract = get_arxiv_abstract(arxiv_id)
            title = get_arxiv_title(arxiv_id)
            text_segments = obj["text"]
            
            # dir = Path(__file__).parent.parent.parent / "relevant_pages" / arxivid
            # dir.mkdir(exist_ok=True,parents=True)
            # with open(dir / f"relevant_pages_{arxiv_id_from_selected_papers}.txt", "w", encoding="utf-8") as f:
            #     f.write("arxiv_id_from_obj: "+arxiv_id+"\n"+"arxiv_id_from_selected_papers: "+arxiv_id_from_selected_papers+"\n"+"arxiv_abstract_from_obj: "+arxiv_abstract+"\n"+"arxiv_abstract_from_selected_papers: "+arxiv_abstract_from_selected_papers+"\n"+"arxiv_title_from_obj: "+title+"\n"+"arxiv_title_from_selected_papers: "+arxiv_title_from_selected_papers+"\n"+"text_segments_from_obj: "+"\n".join(text_segments)+"\n"+"text_segments_from_selected_papers: "+"\n".join(text_segments_from_selected_papers))
            
            # Create prompt
            prompt = generate_summary_prompt_with_page_content(
                abstract_source_paper=abstract,
                abstract_to_be_cited=arxiv_abstract,
                page_text_to_be_cited=text_segments,
                sentence_count=5,
            )
            internal_collection[arxiv_id] = Paper(
                arxiv_id, 
                title, 
                arxiv_abstract, 
                label_opts=["tasks", "datasets", "methodologies", "evaluation_methods"], 
                internal=True
            )
            temp_dict = {"Title": title, "Abstract": arxiv_abstract}
            data.append(temp_dict)
            # Use the appropriate LLM client based on the provider
            response: str = self.llm_client.get_completion(prompt)
            obj["summary"] = response
            obj["citation"] = get_arxiv_citation(arxiv_id)
            
            idx = {
                "cite_ids":[id],
                "title":title,
                "abstract":arxiv_abstract,
                "summary":response,
                "arxiv_id":arxiv_id,
                "citation":get_arxiv_citation(arxiv_id)
            }
            
            
            selected_papers.append(idx)
            id += 1
            dir = Path(__file__).parent.parent.parent / "prompt" / arxivid
            dir.mkdir(exist_ok=True,parents=True)
            with open(dir / f"prompt_{arxiv_id}_2.txt", "w", encoding="utf-8") as f:
                f.write(prompt+"\n"+"="*50+"\n"+"response" +"\n"+"="*50+"\n"+response)

        dir = Path(__file__).parent.parent.parent / "selected_papers" / arxivid
        dir.mkdir(exist_ok=True,parents=True)
        with open(dir / f"selected_papers_{arxivid}_{breadth}_{depth}_{diversity}.json", "w", encoding="utf-8") as f:
            json.dump(selected_papers, f, indent=4, ensure_ascii=False)
            
        if status_callback:
            status_callback(7, "Generated summaries of papers (and their pages)")
            
        # Generate the final related works section text
        prompt = generate_related_work_prompt(
            source_abstract=abstract, data=selected_papers, paragraph_count=math.ceil(breadth / 2), add_summary=False
        )

        # Use the appropriate LLM client based on provider
        related_works_section: str = self.llm_client.get_completion(prompt)
        
        related_works_section = related_works_section.replace("```json","").replace("```","")
        related_works_section = json_repair.loads(related_works_section)
        related_work = related_works_section["related_work"]
        citations = related_works_section["cite_ids"]

        filtered_citations: list[str] = filter_citations(
            related_works_section=related_work, citation_strings=[obj["citation"] for obj in selected_papers]
        )

        if status_callback:
            status_callback(8, f"Generated related work section with {len(filtered_citations)} citations")

        return {"related_works": related_work, "cite_ids": citations, "citations": filtered_citations, "selected_papers": selected_papers}

    def generate_related_work_from_paper(
        self,
        pages: list[str],
        breadth: int,
        depth: int,
        diversity: float,
        status_callback: Optional[Callable] = None,
    ) -> dict[str, str | list[str]]:
        """
        Generate a related work section based on a full paper.

        Args:
            pages: List of paper pages
            breadth: Number of papers to consider
            depth: Number of pages to extract from each paper
            diversity: Diversity factor for paper selection (0-1)
            status_callback: Callback function that will update jobs according to the function progress

        Returns:
            Dictionary with 'related_works' text and 'citations' list
        """
        if status_callback:
            status_callback(1, "Initializing.")

        # Create embeddings for all pages
        page_embeddings = [self.sentence_embedding_model.encode(page) for page in pages]

        # Query Milvus Vector DB for each page
        if status_callback:
            status_callback(2, "Querying Vector DB for matches (this may take a while)")

        all_query_data: list[list[dict]] = []
        for embedding in page_embeddings:
            query_result = self.db_client.search(
                collection_name="abstracts",
                data=[embedding],
                limit=6 * breadth,
                anns_field="embedding",
                # filter = f'topic == {topic_id}',  # Could potentially use topic_ids here
                search_params={"metric_type": "COSINE", "params": {}},
                output_fields=["embedding"],
            )
            all_query_data.extend(query_result)

        if status_callback:
            status_callback(3, f"Retrieved papers from DB for {len(all_query_data)} pages")

        # Aggregate similarity scores for papers that appear multiple times
        paper_scores: dict[str, float] = {}
        paper_data: dict[str, dict] = {}

        for page_results in all_query_data:
            for result in page_results:
                paper_id = result["id"]
                similarity_score = result["distance"]  # Assuming this is the similarity score

                if paper_id in paper_scores:
                    paper_scores[paper_id] += similarity_score
                else:
                    paper_scores[paper_id] = similarity_score
                    paper_data[paper_id] = {"id": paper_id, "embedding": result["entity"]["embedding"]}

        # Convert aggregated results back to format expected by select_diverse_papers
        # Sort papers by aggregated score and take top 6*breadth papers
        top_paper_ids = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)[: 6 * breadth]

        # Convert back to original format expected by select_diverse_papers
        # Each entry should be a list with one dict per query result
        aggregated_query_data = [
            {"id": paper_id, "embedding": paper_data[paper_id]["embedding"], "distance": score}
            for paper_id, score in top_paper_ids
        ]

        # Select a longlist of papers using aggregated scores
        selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
            paper_data=aggregated_query_data, k=3 * breadth, diversity_weight=diversity
        )

        if status_callback:
            status_callback(
                4,
                f"Selected {len(selected_papers)} papers for the longlist, retrieving full text(s)"
                f" (this might take a while)",
            )

        # Generate embeddings of each page of every paper in the longlist
        page_embeddings_papers: list[list[dict]] = []
        for paper in selected_papers:
            arxiv_id = paper["id"]
            result = process_arxiv_paper_with_embeddings(arxiv_id, self.topic_model)
            if result:
                page_embeddings_papers.append(result)

        if status_callback:
            status_callback(5, f"Generated page embeddings for {len(page_embeddings)} papers")

        # Generate shortlist of papers using first page as reference
        # (you might want to modify this to consider all input pages)
        relevant_pages: list[dict] = select_diverse_pages_for_top_b_papers(
            paper_embeddings=page_embeddings_papers,
            input_string=pages[0],  # Using first page as reference
            topic_model=self.topic_model,
            k=depth,
            b=breadth,
            diversity_weight=diversity,
            skip_first=False,
        )

        if status_callback:
            status_callback(6, f"Selected {len(relevant_pages)} papers for the shortlist")

        # Generate summaries for individual papers
        for obj in relevant_pages[1:]:
            arxiv_id = aggregated_query_data[obj["paper_id"]]["id"]
            arxiv_abstract = get_arxiv_abstract(arxiv_id)
            text_segments = obj["text"]
            # Create prompt
            prompt = generate_summary_prompt_with_page_content(
                abstract_source_paper=pages[0],  # Using first page as reference
                abstract_to_be_cited=arxiv_abstract,
                page_text_to_be_cited=text_segments,
                sentence_count=5,
            )

            # Use the appropriate LLM client
            response: str = self.llm_client.get_completion(prompt)
            obj["summary"] = response
            obj["citation"] = get_arxiv_citation(arxiv_id)

        if status_callback:
            status_callback(7, "Generated summaries of papers (and their pages)")

        # Generate the final related works section text
        prompt = generate_related_work_prompt(
            source_abstract=pages[0],  # Using first page as reference
            data=relevant_pages,
            paragraph_count=math.ceil(breadth / 2),
            add_summary=False,
        )

        # Use the appropriate LLM client
        related_works_section: str = self.llm_client.get_completion(prompt)

        filtered_citations: list[str] = filter_citations(
            related_works_section=related_works_section, citation_strings=[obj["citation"] for obj in relevant_pages]
        )

        if status_callback:
            status_callback(8, f"Generated related work section with {len(filtered_citations)} citations")

        return {"related_works": related_works_section, "citations": filtered_citations}

    def generate_answer_to_scientific_question(
        self,
        question: str,
        breadth: int,
        depth: int,
        diversity: float,
        status_callback: Optional[Callable] = None,
    ) -> dict[str, str | list[str]]:
        """
        Generate an answer to a scientific question.

        Args:
            question: The input question text
            breadth: Number of papers to consider
            depth: Number of pages to extract from each paper
            diversity: Diversity factor for paper selection (0-1)
            status_callback: Callback function that will update jobs according to the function progress

        Returns:
            Dictionary with 'question_answer' text and 'citations' list
        """
        if status_callback:
            status_callback(1, "Initializing.")

        embedded_abstract = self.sentence_embedding_model.encode(question)
        # topic = self.topic_model.transform(question)
        # topic_id = topic[0][0]

        # Query Milvus Vector DB
        if status_callback:
            status_callback(2, "Querying Vector DB for matches (this may take a while)")

        query_data: list[list[dict]] = self.db_client.search(
            collection_name="abstracts",
            data=[embedded_abstract],
            limit=6 * breadth,
            anns_field="embedding",
            # filter = f'topic == {topic_id}',
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["embedding"],
        )

        if status_callback:
            status_callback(3, f"Retrieved {len(query_data[0])} papers from the DB")

        # Clean DB response data
        papers_data: list[dict] = query_data[0]
        for obj in papers_data:
            obj["embedding"] = obj["entity"]["embedding"]
            obj.pop("entity")

        # Select a longlist of papers
        selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
            paper_data=papers_data, k=3 * breadth, diversity_weight=diversity
        )

        if status_callback:
            status_callback(
                4,
                f"Selected {len(selected_papers)} papers for the longlist, retrieving full text(s)"
                f" (this might take a while)",
            )

        # Generate embeddings of each page of every paper in the longlist
        page_embeddings: list[list[dict]] = []
        for paper in selected_papers:
            arxiv_id = paper["id"]
            result = process_arxiv_paper_with_embeddings(arxiv_id, self.topic_model)
            if result:
                page_embeddings.append(result)

        if status_callback:
            status_callback(5, f"Generated page embeddings for {len(page_embeddings)} papers")

        # Generate shortlist of papers (at most k pages per paper, at most b papers in total)
        relevant_pages: list[dict] = select_diverse_pages_for_top_b_papers(
            paper_embeddings=page_embeddings,
            input_string=question,
            topic_model=self.topic_model,
            k=depth,
            b=breadth,
            diversity_weight=diversity,
            skip_first=False,
        )

        if status_callback:
            status_callback(6, f"Selected {len(relevant_pages)} papers for the shortlist")

        # Generate summaries for individual papers (taking all relevant pages into account)
        for obj in relevant_pages[1:]:
            # Because paper_id != arXiv_id -> retrieve arXiv id/
            arxiv_id = papers_data[obj["paper_id"]]["id"]
            arxiv_abstract = get_arxiv_abstract(arxiv_id)
            text_segments = obj["text"]
            # Create prompt
            prompt = generate_summary_prompt_question_with_page_content(
                question=question, abstract_to_be_considered=arxiv_abstract, page_text_to_be_cited=text_segments
            )

            # Use the appropriate LLM client
            response: str = self.llm_client.get_completion(prompt)
            obj["summary"] = response
            obj["citation"] = get_arxiv_citation(arxiv_id)

        if status_callback:
            status_callback(7, "Generated summaries of papers (and their pages)")

        # Generate the final question answer
        prompt = generate_question_answer_prompt(question=question, data=relevant_pages)

        # Use the appropriate LLM client
        question_answer: str = self.llm_client.get_completion(prompt)

        filtered_citations: list[str] = filter_citations(
            related_works_section=question_answer, citation_strings=[obj["citation"] for obj in relevant_pages]
        )

        if status_callback:
            status_callback(8, f"Generated answer to question with {len(filtered_citations)} citations")

        return {"question_answer": question_answer, "citations": filtered_citations}