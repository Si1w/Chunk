import os
import argparse
from git import Repo
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def clone_repo(repo, root_dir, token):
    """
    Clone a GitHub repository to the specified directory.
    """
    repo_dir = Path(root_dir, f"{repo.replace('/', '__')}")

    if not repo_dir.exists():
        repo_url = f"https://{token}@github.com/{repo}.git"
        logger.info(f"Cloning {repo}")
        Repo.clone_from(repo_url, repo_dir)
    return repo_dir

def build_repo_rels(dataset="princeton-nlp/SWE-bench_Lite", split="dev", root_dir="./eval/swebench/repos", token=None):
    """
    Build relations from dataset.
    """

    os.makedirs(root_dir, exist_ok=True)

    logger.info(f"Loading {dataset} {split} split...")
    dataset = load_dataset(dataset, split=split)
    
    for instance in tqdm(dataset, desc="Processing instances"):
        # if instance["repo"] == "pvlib/pvlib-python" or instance["repo"] == "pydicom/pydicom":
        #     continue
        repo = instance["repo"]
        instance_id = instance["instance_id"]
        
        try:
            clone_repo(repo, root_dir, token)
        except Exception as e:
            logger.error(f"Failed to process {instance_id}: {e}")
            continue

def main(): 
    parser = argparse.ArgumentParser(description="Build repo rels from datasets")
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--root_dir", type=str, default="./eval/swebench/repos")
    parser.add_argument("--token", type=str, default=None)
    
    args = parser.parse_args()
    
    token = args.token or os.environ.get("GITHUB_TOKEN")
    
    if not token:
        logger.warning("No GitHub token provided. Rate limiting may occur.")

    build_repo_rels(
        dataset=args.dataset,
        split=args.split,
        root_dir=args.root_dir,
        token=token
    )

if __name__ == "__main__":
    main()