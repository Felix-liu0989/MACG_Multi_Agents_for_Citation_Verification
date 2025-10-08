from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
arxiv_id = "2410.0333"
results_dir = PROJECT_ROOT / f"selected_papers/{arxiv_id.replace('.','_')}"

results_dir.mkdir(exist_ok=True,parents=True)