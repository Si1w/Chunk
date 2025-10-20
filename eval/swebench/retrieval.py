import json
import argparse
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

class EmbeddingRetriever:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        logger.info(f"Loading model {model_name} on {self.device}")
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        logger.info("Model loaded successfully")

    def retrieve(self, query: str, documents: list[str], top_k: int):
        """
        Retrieve top-k most relevant documents for the query.
        Process documents one by one to save memory.
        
        Args:
            query (str): Query text
            documents (list): List of document texts
            top_k (int): Number of top documents to retrieve
            
        Returns:
            list: List of tuples (index, score, document)
        """
        logger.info(f"Encoding query...")
        with torch.no_grad():
            query_embeddings = self.model.encode(query, prompt_name="query")
            document_embeddings = self.model.encode(documents, batch_size=4, show_progress_bar=True)
        similarity = self.model.similarity(query_embeddings, document_embeddings)

        # Get top-k results
        similarity = similarity.cpu().numpy().ravel()
        top_k_retrieval = min(top_k, len(documents))
        top_k_indices = np.argsort(similarity)[-top_k_retrieval:][::-1]
        results = []
        for idx in top_k_indices:
            results.append((idx, float(similarity[idx]), documents[idx]))
        return results

def load_corpus(corpus_path: str):
        """
        Load corpus from JSON file.
        """
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        return corpus

def prepare_documents(corpus):
    """
    Prepare documents from corpus for retrieval.
    Returns list of documents and their metadata.
    """
    documents = []
    doc_metadata = []

    for file_path, chunks in corpus['documents'].items():
        if isinstance(chunks, str):
            if chunks.strip():
                documents.append(chunks)
                doc_metadata.append({
                    "file_path": file_path,
                    "chunk_index": 0,
                    "metadata": {}
                })
            continue

        if isinstance(chunks, list):
            for idx, chunk in enumerate(chunks):
                text = None
                meta = {}
                if isinstance(chunk, dict):
                    if 'text' in chunk and isinstance(chunk['text'], str):
                        text = chunk['text']
                    elif 'content' in chunk and isinstance(chunk['content'], str):
                        text = chunk['content']
                    meta = chunk.get('metadata', {})
                elif isinstance(chunk, str):
                    text = chunk

                if not text or not text.strip():
                    continue

                documents.append(text)
                doc_metadata.append({
                    "file_path": file_path,
                    "chunk_index": idx,
                    "metadata": meta,
                })

    return documents, doc_metadata

def run_retrieval_process(dataset, retriever, corpus_dir, method, top_k):
    """
    Run retrieval for all instances with a single chunking method.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing with {method} chunking method")
    logger.info(f"{'='*60}\n")
    retrieved_docs = {}
    for instance in tqdm(dataset, desc=f"Retrieval with {method}"):
        if instance["repo"] == "pvlib/pvlib-python" or instance["repo"] == "pydicom/pydicom":
            continue
        instance_id = instance["instance_id"]
        problem_statement = instance["problem_statement"]

        corpus_path = Path(corpus_dir) / f"{method}_corpus.json"

        if not corpus_path.exists():
            logger.warning(f"Corpus file not found for {instance_id} with {method}")
            continue

        try:
            corpus = load_corpus(str(corpus_path))
            documents, doc_metadata = prepare_documents(corpus[instance_id])

            if not documents:
                logger.warning(f"No documents found for {instance_id} with {method}")
                continue

            logger.info(f"Retrieving for {instance_id} ({len(documents)} documents)")

            results = retriever.retrieve(problem_statement, documents, top_k)

            retrieval_results = {
                "query": problem_statement,
                "method": method,
                "top_k": top_k,
                "results": []
            }

            for rank, (doc_idx, score, doc_text) in enumerate(results, start=1):
                metadata = doc_metadata[doc_idx]
                result_entry = {
                    "rank": rank,
                    "score": score,
                    "content": doc_text,
                    "file_path": metadata.get("file_path"),
                    "chunk_index": metadata.get("chunk_index"),
                    "metadata": metadata.get("metadata", {}),
                }
                retrieval_results["results"].append(result_entry)
            retrieved_docs[instance_id] = retrieval_results

        except Exception as e:
            logger.error(f"Failed to process {instance_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return retrieved_docs

def run_retrieval(model: str, dataset_name: str, split: str, corpus_dir: str, output_dir: str, chunking_method: str, top_k: int, device: str):
    """
    Run retrieval for all instances in the dataset.
    """
    logger.info(f"Loading {dataset_name} {split} split...")
    dataset = load_dataset(dataset_name, split=split)

    retriever = EmbeddingRetriever(model_name=model, device=device)
    
    methods_to_process = []
    if chunking_method == "all":
        methods_to_process = ["sliding", "function", "hierarchical", "cAST"]
        # methods_to_process = ["function", "hierarchical", "cAST"]
    else:
        methods_to_process = [chunking_method]

    for method in methods_to_process:
        retrieved_docs = run_retrieval_process(
            dataset=dataset,
            retriever=retriever,
            corpus_dir=corpus_dir,
            method=method,
            top_k=top_k,
        )
        model_name_safe = model.split("/")[-1]
        output_folder = Path(output_dir) / model_name_safe
        output_folder.mkdir(parents=True, exist_ok=True)

        output_path = output_folder / f"{method}_retrieval.json"
        with open(str(output_path), 'w', encoding='utf-8') as f:
            json.dump(retrieved_docs, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved to {output_path}")
    logger.info("All retrieval completed successfully")

def main():
    parser = argparse.ArgumentParser(description="Retrieve documents using Embedding Model")
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--corpus-dir", type=str, default="./eval/swebench/corpus")
    parser.add_argument("--output-dir", type=str, default="./eval/swebench/retrieval")
    parser.add_argument("--method", type=str, default="all",
                        choices=["sliding", "function", "hierarchical", "cAST", "all"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="HuggingFace model id for embedding")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    run_retrieval(
        model=args.model,
        dataset_name=args.dataset,
        split=args.split,
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        chunking_method=args.method,
        top_k=args.top_k,
        device=args.device
    )

if __name__ == "__main__":
    main()