"""
Agent通信上下文分析工具
用于分析和可视化agent之间的信息流动和损失情况
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class InformationFlowAnalysis:
    """信息流动分析结果"""
    agent_name: str
    input_info: Dict[str, int]  # 信息类型 -> 信息量(字符数或项数)
    output_info: Dict[str, int]
    information_loss: Dict[str, int]  # 丢失的信息
    information_gain: Dict[str, int]  # 新增的信息
    
    def get_loss_percentage(self) -> float:
        """计算信息损失百分比"""
        total_input = sum(self.input_info.values())
        total_loss = sum(self.information_loss.values())
        if total_input == 0:
            return 0.0
        return (total_loss / total_input) * 100


class ContextAnalyzer:
    """上下文分析器"""
    
    @staticmethod
    def analyze_information_flow(
        agent_history: List[Dict[str, Any]]
    ) -> List[InformationFlowAnalysis]:
        """分析agent执行历史中的信息流动"""
        analyses = []
        
        for i, record in enumerate(agent_history):
            agent_name = record["agent_name"]
            input_summary = record["input_summary"]
            output_summary = record["output_summary"]
            
            # 分析输入信息
            input_info = ContextAnalyzer._parse_summary(input_summary)
            output_info = ContextAnalyzer._parse_summary(output_summary)
            
            # 识别信息损失和增益
            information_loss = {}
            information_gain = {}
            
            # 检查每个输入信息项
            for key, value in input_info.items():
                if key not in output_info:
                    information_loss[key] = value
                elif output_info[key] < value:
                    information_loss[key] = value - output_info[key]
            
            # 检查新增信息
            for key, value in output_info.items():
                if key not in input_info:
                    information_gain[key] = value
                elif output_info[key] > input_info[key]:
                    information_gain[key] = output_info[key] - input_info[key]
            
            analysis = InformationFlowAnalysis(
                agent_name=agent_name,
                input_info=input_info,
                output_info=output_info,
                information_loss=information_loss,
                information_gain=information_gain
            )
            analyses.append(analysis)
        
        return analyses
    
    @staticmethod
    def _parse_summary(summary: Dict[str, Any]) -> Dict[str, int]:
        """解析summary,提取信息量"""
        info = {}
        for key, value in summary.items():
            if isinstance(value, str):
                if value.startswith("List["):
                    # 提取列表长度
                    count = int(value.split("[")[1].split(" ")[0])
                    info[key] = count
                elif value.startswith("Dict["):
                    # 提取字典键数
                    count = int(value.split("[")[1].split(" ")[0])
                    info[key] = count
                elif "..." in value:
                    # 截断的文本,估算长度
                    info[key] = 100  # 假设原文本>100字符
                else:
                    info[key] = len(value)
            elif isinstance(value, (int, float)):
                info[key] = int(value)
        return info
    
    @staticmethod
    def compare_workflows(
        original_result: Dict[str, Any],
        optimized_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """对比优化前后的工作流结果"""
        comparison = {
            "validation_metrics_improvement": {},
            "information_preservation": {},
            "execution_efficiency": {}
        }
        
        # 对比验证指标
        original_validation = original_result.get("validation_results", {})
        optimized_validation = optimized_result.get("validation_results", {})
        
        for metric in ["claim_precision", "citation_precision", "reference_precision"]:
            orig_val = original_validation.get(metric, 0)
            opt_val = optimized_validation.get(metric, 0)
            improvement = ((opt_val - orig_val) / orig_val * 100) if orig_val > 0 else 0
            comparison["validation_metrics_improvement"][metric] = {
                "original": orig_val,
                "optimized": opt_val,
                "improvement_percent": round(improvement, 2)
            }
        
        # 对比信息保留
        original_info_items = ContextAnalyzer._count_information_items(original_result)
        optimized_info_items = ContextAnalyzer._count_information_items(optimized_result)
        
        comparison["information_preservation"] = {
            "original_items": original_info_items,
            "optimized_items": optimized_info_items,
            "additional_items_preserved": optimized_info_items - original_info_items
        }
        
        # 对比执行效率
        original_time = original_result.get("time_dict", {}).get("all_time", 0)
        optimized_time = optimized_result.get("time_dict", {}).get("all_time", 0)
        time_diff = optimized_time - original_time
        
        comparison["execution_efficiency"] = {
            "original_time": original_time,
            "optimized_time": optimized_time,
            "time_difference": round(time_diff, 2),
            "time_increase_percent": round((time_diff / original_time * 100) if original_time > 0 else 0, 2)
        }
        
        return comparison
    
    @staticmethod
    def _count_information_items(result: Dict[str, Any]) -> int:
        """统计结果中的信息项数量"""
        count = 0
        
        # 计算papers的信息项
        if "selected_papers" in result:
            papers = result["selected_papers"]
            for paper in papers:
                count += len(paper.keys())
                # 如果有full_text_segments,额外计分
                if "full_text_segments" in paper:
                    count += len(paper["full_text_segments"])
        
        # 计算其他字段
        if "related_works" in result:
            count += 1
        if "citations" in result:
            count += len(result["citations"])
        if "validation_results" in result:
            count += len(result["validation_results"])
        
        return count
    
    @staticmethod
    def generate_report(
        agent_history: List[Dict[str, Any]],
        comparison: Dict[str, Any] = None
    ) -> str:
        """生成分析报告"""
        report = []
        report.append("=" * 80)
        report.append("Agent通信分析报告")
        report.append("=" * 80)
        
        # 分析信息流动
        analyses = ContextAnalyzer.analyze_information_flow(agent_history)
        
        report.append("\n## 信息流动分析\n")
        for i, analysis in enumerate(analyses, 1):
            report.append(f"\n### {i}. {analysis.agent_name}")
            report.append(f"   输入信息: {analysis.input_info}")
            report.append(f"   输出信息: {analysis.output_info}")
            
            if analysis.information_loss:
                loss_pct = analysis.get_loss_percentage()
                report.append(f"   ⚠️  信息损失: {analysis.information_loss} ({loss_pct:.1f}%)")
            else:
                report.append(f"   ✓  无信息损失")
            
            if analysis.information_gain:
                report.append(f"   ✓  新增信息: {analysis.information_gain}")
        
        # 总体信息损失统计
        total_loss_agents = sum(1 for a in analyses if a.information_loss)
        avg_loss_pct = sum(a.get_loss_percentage() for a in analyses) / len(analyses) if analyses else 0
        
        report.append("\n## 总体统计\n")
        report.append(f"   总agent数: {len(analyses)}")
        report.append(f"   存在信息损失的agent数: {total_loss_agents}")
        report.append(f"   平均信息损失率: {avg_loss_pct:.1f}%")
        
        # 对比报告
        if comparison:
            report.append("\n" + "=" * 80)
            report.append("优化效果对比")
            report.append("=" * 80)
            
            report.append("\n## 验证指标改进\n")
            for metric, data in comparison["validation_metrics_improvement"].items():
                report.append(f"\n   {metric}:")
                report.append(f"      原始: {data['original']:.3f}")
                report.append(f"      优化后: {data['optimized']:.3f}")
                improvement = data['improvement_percent']
                symbol = "↑" if improvement > 0 else ("↓" if improvement < 0 else "→")
                report.append(f"      改进: {symbol} {abs(improvement):.1f}%")
            
            report.append("\n## 信息保留改进\n")
            info_pres = comparison["information_preservation"]
            report.append(f"   原始工作流信息项: {info_pres['original_items']}")
            report.append(f"   优化后信息项: {info_pres['optimized_items']}")
            report.append(f"   额外保留: +{info_pres['additional_items_preserved']} 项")
            
            report.append("\n## 执行效率\n")
            exec_eff = comparison["execution_efficiency"]
            report.append(f"   原始执行时间: {exec_eff['original_time']:.2f}秒")
            report.append(f"   优化后执行时间: {exec_eff['optimized_time']:.2f}秒")
            time_inc = exec_eff['time_increase_percent']
            if time_inc > 0:
                report.append(f"   时间增加: +{time_inc:.1f}% (由于处理更多信息)")
            elif time_inc < 0:
                report.append(f"   时间减少: {time_inc:.1f}%")
            else:
                report.append(f"   时间持平")
        
        report.append("\n" + "=" * 80)
        report.append("报告结束")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    @staticmethod
    def visualize_information_flow(
        agent_history: List[Dict[str, Any]],
        output_file: Path
    ):
        """可视化信息流动(生成Mermaid图表)"""
        analyses = ContextAnalyzer.analyze_information_flow(agent_history)
        
        mermaid = ["graph TD"]
        
        # 创建节点
        for i, analysis in enumerate(analyses):
            node_id = f"A{i}"
            label = f"{analysis.agent_name}"
            
            # 添加信息损失标记
            if analysis.information_loss:
                loss_pct = analysis.get_loss_percentage()
                label += f"\\n⚠️ -{loss_pct:.0f}%"
            
            mermaid.append(f'    {node_id}["{label}"]')
        
        # 创建连接
        for i in range(len(analyses) - 1):
            current = f"A{i}"
            next_node = f"A{i+1}"
            
            # 检查传递的信息
            current_output = analyses[i].output_info
            next_input = analyses[i+1].input_info
            
            # 找出共同的信息项
            common_keys = set(current_output.keys()) & set(next_input.keys())
            lost_keys = set(current_output.keys()) - set(next_input.keys())
            
            if lost_keys:
                label = f"丢失: {', '.join(list(lost_keys)[:2])}"
                mermaid.append(f'    {current} -->|{label}| {next_node}')
            else:
                mermaid.append(f'    {current} --> {next_node}')
        
        # 保存
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(mermaid))
        
        # 同时保存为markdown以便查看
        md_file = output_file.with_suffix(".md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# Agent信息流动图\n\n")
            f.write("```mermaid\n")
            f.write("\n".join(mermaid))
            f.write("\n```\n")


def analyze_results_directory(results_dir: Path) -> Dict[str, Any]:
    """分析results目录中的结果文件"""
    
    # 查找所有结果文件
    original_files = list(results_dir.glob("results_*.json"))
    context_files = list(results_dir.glob("agent_context_*.json"))
    
    analysis = {
        "original_results": [],
        "optimized_results": [],
        "comparisons": []
    }
    
    # 加载原始结果
    for file in original_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            analysis["original_results"].append({
                "file": str(file),
                "data": data
            })
    
    # 加载优化后的结果(带上下文的)
    for file in context_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            analysis["optimized_results"].append({
                "file": str(file),
                "data": data
            })
    
    # 如果有成对的结果,进行对比
    if analysis["original_results"] and analysis["optimized_results"]:
        for orig in analysis["original_results"]:
            for opt in analysis["optimized_results"]:
                # 检查是否是同一个实验的结果(通过参数匹配)
                # 这里简化处理,实际可以通过文件名或参数判断
                comparison = ContextAnalyzer.compare_workflows(
                    orig["data"],
                    opt["data"]
                )
                analysis["comparisons"].append({
                    "original_file": orig["file"],
                    "optimized_file": opt["file"],
                    "comparison": comparison
                })
    
    return analysis


if __name__ == "__main__":
    # 示例: 分析agent执行历史
    mock_history = [
        {
            "agent_name": "dag_builder",
            "timestamp": 1.5,
            "input_summary": {"dimensions": 2, "papers": "List[10 items]"},
            "output_summary": {"grouped_papers": "Dict[2 keys]"},
            "metadata": {}
        },
        {
            "agent_name": "related_work_generator",
            "timestamp": 2.3,
            "input_summary": {"grouped_papers": "Dict[2 keys]"},
            "output_summary": {"related_work": "This is a long text...", "citations": "List[5 items]"},
            "metadata": {}
        },
        {
            "agent_name": "feedback_agent",
            "timestamp": 1.8,
            "input_summary": {"related_work": "This is a long text..."},
            "output_summary": {"feedback": "The related work needs..."},
            "metadata": {}
        }
    ]
    
    report = ContextAnalyzer.generate_report(mock_history)
    print(report)


