import os
import json
import argparse
from astchunk import ASTChunkBuilder
from git import Repo
from pathlib import Path
from datasets import load_dataset
from swebench.inference.make_datasets.utils import list_files
from src.methods import SlidingWindowChunk, FunctionLevelChunk, HierarchicalChunk, NaturalBoundaryChunk
from tqdm import tqdm

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

class ContextManager:
    """
    A context manager for managing a Git repository at a specific commit.
    """

    def __init__(self, repo_path, base_commit, verbose=False):
        self.repo_path = Path(repo_path).resolve().as_posix()
        self.base_commit = base_commit
        self.verbose = verbose
        self.repo = Repo(self.repo_path)

    def __enter__(self):
        if self.verbose:
            print(f"Switching to {self.base_commit}")
        try:
            self.repo.git.reset("--hard", self.base_commit)
            self.repo.git.clean("-fdxq")
        except Exception as e:
            logger.error(f"Failed to switch to {self.base_commit}")
            logger.error(e)
            raise e
        return self
    
    def get_readme_files(self):
        files = os.listdir(self.repo_path)
        files = list(filter(lambda x: os.path.isfile(x), files))
        files = list(filter(lambda x: x.lower().startswith("readme"), files))
        return files

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def chunkify(repo_dir: str, base_commit: str, chunker):
    """
    Apply chunking method to all files in the repository at a specific commit.
    """
    documents = {}
    with ContextManager(repo_dir, base_commit):
        filenames = list_files(repo_dir, include_tests=False)
        for relative_path in filenames:
            filename = os.path.join(repo_dir, relative_path)
            
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # Apply chunking for Python files
                if filename.endswith('.py'):
                    chunks = chunker.chunkify(code, repo_level_metadata={"filepath": relative_path})
                    documents[relative_path] = chunks
                else:
                    documents[relative_path] = code
            except Exception as e:
                logger.warning(f"Failed to process {relative_path}: {e}")
                continue
    return documents

def process_repos(dataset_name: str, split: str, repos_dir: str, output_dir: str, chunking_method: str, **configs):
    """
    Process all repositories from SWE-bench Lite with the specified chunking method.
    Create a separate folder for each instance_id with its corpus.
    """
    logger.info(f"Loading {dataset_name} {split} split...")
    dataset = load_dataset(dataset_name, split=split)
    
    methods_to_process = []
    if chunking_method == "all":
        methods_to_process = ["sliding", "function", "hierarchical", "cAST", "natural-boundary"]
    else:
        methods_to_process = [chunking_method]
    
    for method in methods_to_process:
        chunk = {}
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing with {method} chunking method")
        logger.info(f"{'='*60}\n")
        
        if method == "sliding":
            chunker = SlidingWindowChunk(**configs)
        elif method == "function":
            chunker = FunctionLevelChunk(**configs)
        elif method == "hierarchical":
            chunker = HierarchicalChunk(**configs)
        elif method == "cAST":
            chunker = ASTChunkBuilder(**configs)
        elif method == "natural-boundary":
            chunker = NaturalBoundaryChunk(**configs)
        else:
            raise ValueError(f"Unknown chunking method: {method}")
        
        for instance in tqdm(dataset, desc=f"Processing with {method}"):
            # if instance["repo"] == "pvlib/pvlib-python" or instance["repo"] == "pydicom/pydicom":
            #     continue
            repo = instance["repo"]
            base_commit = instance["base_commit"]
            instance_id = instance["instance_id"]
            repo_dir = Path(repos_dir, f"{repo.replace('/', '__')}")
            
            if not repo_dir.exists():
                logger.warning(f"Repository {repo} not found at {repo_dir}")
                continue
            
            try:
                logger.info(f"Processing {instance_id} at {base_commit[:8]}")
                documents = chunkify(str(repo_dir), base_commit, chunker)
                os.makedirs(output_dir, exist_ok=True)
                
                corpus = {
                    "repo": repo,
                    "base_commit": base_commit,
                    "documents": documents
                }
                chunk[instance_id] = corpus

            except Exception as e:
                logger.error(f"Failed to process {instance_id}: {e}")
                continue
        
        output_file = Path(output_dir, f"{method}_corpus.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, indent=2)
        logger.info(f"Saved {method} chunked corpus to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Chunkify repositories from SWE-bench Lite")
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--repos_dir", type=str, default="./eval/swebench/repos")
    parser.add_argument("--output_dir", type=str, default="./eval/swebench/corpus")
    parser.add_argument("--method", type=str, default="all", choices=["sliding", "function", "hierarchical", "cAST", "natural-boundary", "all"])
    parser.add_argument("--max_chunk_size", type=int, default=500)
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument("--metadata_template", type=str, default="default")
    parser.add_argument("--chunk-expansion", action="store_true")
    parser.add_argument("--overlap_lines", type=int, default=100)
    
    args = parser.parse_args()
    
    configs = {
        "max_chunk_size": args.max_chunk_size,
        "language": args.language,
        "metadata_template": args.metadata_template,
        "chunk_expansion": args.chunk_expansion,
        "overlap_lines": args.overlap_lines
    }
    
    process_repos(
        dataset_name=args.dataset,
        split=args.split,
        repos_dir=args.repos_dir,
        output_dir=args.output_dir,
        chunking_method=args.method,
        **configs
    )

if __name__ == "__main__":
    main()