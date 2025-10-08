# Deep Research Agents 评估系统

基于MACG框架的Deep Research Agents评估系统，专门用于评测deep research agents在引用可靠性和事实支持方面的表现。

## 系统概述

本评估系统扩展了原有的MACG框架，增加了针对deep research agents的专门评估指标和方法。系统可以：

1. **引用可靠性评估** - 评估引用的准确性、相关性、时效性和完整性
2. **事实支持评估** - 评估事实的准确性、可验证性、一致性和完整性
3. **基线对比** - 与现有baseline方法进行对比
4. **批量评估** - 支持大规模批量评估
5. **详细分析** - 提供详细的错误分析和改进建议

## 核心组件

### 1. DeepResearchEvaluator
主要的评估器，包含以下评估维度：

**引用可靠性指标：**
- `citation_accuracy`: 引用准确性
- `citation_relevance`: 引用相关性  
- `citation_timeliness`: 引用时效性
- `citation_completeness`: 引用完整性
- `citation_density`: 引用密度
- `citation_coverage`: 引用覆盖度

**事实支持指标：**
- `factual_accuracy`: 事实准确性
- `factual_verifiability`: 事实可验证性
- `factual_consistency`: 事实一致性
- `factual_completeness`: 事实完整性
- `hallucination_rate`: 幻觉率

### 2. BaselineComparisonEvaluator
基线对比评估器，支持与以下方法对比：
- MACG (Multi-Agent Collaborative Generation)
- Perplexity Deep Research
- Naive RAG with GPT
- GPT-4o Mini Search

### 3. DeepResearchEvaluationPipeline
完整的评估流水线，支持：
- 单个摘要评估
- 批量评估
- 文件输入评估
- 自动报告生成

## 安装和配置

### 1. 环境要求
```bash
# 创建conda环境
conda create -n deep_research_eval python=3.10 -y
conda activate deep_research_eval

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境变量配置
创建 `.env` 文件：
```bash
# LLM API Keys
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key
DEEPSEEK_API_KEY=your_deepseek_key

# Vector Database
MILVUS_URI=http://your_milvus_uri:19530
MILVUS_TOKEN=your_milvus_token
```

### 3. 配置文件
使用 `evaluation/config.json` 进行详细配置，包括：
- LLM提供商设置
- 评估参数
- 指标权重
- 输出格式

## 使用方法

### 1. 基本使用

```python
from evaluation.deep_research_evaluation_pipeline import DeepResearchEvaluationPipeline, EvaluationConfig

# 创建配置
config = EvaluationConfig(
    llm_provider="gemini",
    breadth=10,
    depth=2,
    diversity=0.0,
    enable_baseline_comparison=True
)

# 创建评估流水线
pipeline = DeepResearchEvaluationPipeline(config)

# 评估单个摘要
abstract = "Your research abstract here..."
result = pipeline.evaluate_single_abstract(abstract, "paper_id")
```

### 2. 批量评估

```python
# 准备摘要数据
abstracts = [
    {"paper_id": "paper_1", "abstract": "Abstract 1..."},
    {"paper_id": "paper_2", "abstract": "Abstract 2..."},
    # ...
]

# 执行批量评估
results = pipeline.evaluate_batch(abstracts, "output.json")
```

### 3. 从文件评估

```python
# 从JSON文件读取摘要
results = pipeline.evaluate_from_file("input.json", "output.json")
```

### 4. 基线对比

```python
from evaluation.baseline_comparison import BaselineComparisonEvaluator

# 创建对比评估器
comparator = BaselineComparisonEvaluator()

# 执行综合对比
comparison_result = comparator.comprehensive_comparison(abstract)
```

### 5. 命令行使用

```bash
# 基本评估
python evaluation/deep_research_evaluation_pipeline.py -i input.json -o output.json

# 自定义参数
python evaluation/deep_research_evaluation_pipeline.py \
    -i input.json \
    -o output.json \
    --breadth 15 \
    --depth 3 \
    --diversity 0.2 \
    --llm-provider gemini

# 禁用基线对比
python evaluation/deep_research_evaluation_pipeline.py \
    -i input.json \
    -o output.json \
    --no-baseline
```

## 评估指标详解

### 引用可靠性指标

1. **引用准确性 (Citation Accuracy)**
   - 评估引用格式是否正确
   - 检查引用是否真实存在
   - 权重：0.3

2. **引用相关性 (Citation Relevance)**
   - 评估引用是否与声称的内容相关
   - 使用LLM判断相关性
   - 权重：0.3

3. **引用时效性 (Citation Timeliness)**
   - 评估引用是否是最新的
   - 检查引用年份
   - 权重：0.2

4. **引用完整性 (Citation Completeness)**
   - 评估引用信息是否完整
   - 检查是否包含作者、年份、标题等
   - 权重：0.2

### 事实支持指标

1. **事实准确性 (Factual Accuracy)**
   - 评估声称的事实是否准确
   - 使用双模型验证
   - 权重：0.3

2. **事实可验证性 (Factual Verifiability)**
   - 评估事实是否可以从引用中验证
   - 检查引用支持
   - 权重：0.3

3. **事实一致性 (Factual Consistency)**
   - 评估不同引用之间的事实是否一致
   - 检查内部一致性
   - 权重：0.2

4. **事实完整性 (Factual Completeness)**
   - 评估是否遗漏了重要事实
   - 检查覆盖度
   - 权重：0.1

5. **幻觉率 (Hallucination Rate)**
   - 评估无法验证的信息比例
   - 识别幻觉内容
   - 权重：0.1

## 输出格式

### 1. JSON输出
```json
{
  "paper_id": "paper_1",
  "abstract": "Abstract text...",
  "timestamp": "2024-01-01T00:00:00",
  "evaluation_result": {
    "citation_reliability": {
      "citation_accuracy": 0.85,
      "citation_relevance": 0.78,
      "citation_timeliness": 0.92,
      "citation_completeness": 0.88,
      "citation_density": 0.65,
      "citation_coverage": 0.72
    },
    "factual_support": {
      "factual_accuracy": 0.82,
      "factual_verifiability": 0.79,
      "factual_consistency": 0.85,
      "factual_completeness": 0.76,
      "hallucination_rate": 0.15
    },
    "overall_score": 0.81
  },
  "recommendations": [
    "提高引用相关性：优化论文选择和匹配算法",
    "增加引用密度：在更多句子中添加引用"
  ]
}
```

### 2. Markdown报告
自动生成可读的Markdown报告，包含：
- 评估配置
- 统计摘要
- 分数分布
- 改进建议
- 元数据

## 高级功能

### 1. 自定义评估指标
可以通过修改配置文件来自定义指标权重和阈值。

### 2. 错误分析
系统会自动分析常见错误类型：
- 引用格式错误
- 事实准确性错误
- 一致性问题
- 覆盖度问题

### 3. 改进建议
基于评估结果自动生成改进建议：
- 系统性问题识别
- 具体改进方向
- 最佳实践推荐

### 4. 可视化支持
支持生成评估结果的可视化图表（需要额外配置）。

## 性能优化

### 1. 并行处理
支持多线程并行评估以提高效率。

### 2. 缓存机制
支持中间结果缓存以避免重复计算。

### 3. 增量评估
支持断点续传和增量评估。

## 故障排除

### 1. 常见问题
- API密钥配置错误
- 数据库连接问题
- 内存不足
- 网络超时

### 2. 调试模式
启用详细日志记录：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 3. 错误处理
系统包含完善的错误处理机制，会自动记录和报告错误。

## 贡献指南

欢迎贡献代码和建议！请遵循以下步骤：

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 创建Issue
- 发送邮件
- 参与讨论

---

**注意**: 使用本系统需要有效的API密钥和数据库连接。请确保在开始评估前正确配置所有必要的环境变量。

