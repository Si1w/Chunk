import json
import argparse
import os
import numpy as np
from getpass import getpass
from mistralai import Mistral
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

api_key = getpass("Enter your MISTRAL_API_KEY: ").strip()
os.environ["MISTRAL_API_KEY"] = api_key
client = Mistral(api_key=api_key.strip())

class EmbeddingRetriever:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        logger.info(f"Loading model {model_name} on {self.device}")
        self.model = "codestral-embed"
        logger.info("Model loaded successfully")

    def retrieve(self, query: str, documents: list[str], top_k: int, batch_size: int = 16):
        """
        Retrieve top-k most relevant documents for the query.
        Process documents in batches to handle API token limits.
        
        Args:
            query (str): Query text
            documents (list): List of document texts
            top_k (int): Number of top documents to retrieve
            
        Returns:
            list: List of tuples (index, score, document)
        """
        logger.info(f"Encoding query...")
        query_response = client.embeddings.create(
            model=self.model,
            inputs=[query]
        )
        query_embedding = np.array(query_response.data[0].embedding)
        all_embeddings = []
        
        logger.info(f"Encoding {len(documents)} documents in batches of {batch_size}...")
        for i in tqdm(range(0, len(documents), batch_size), desc="Encoding batches"):
            batch = documents[i:i + batch_size]
            try:
                batch_response = client.embeddings.create(
                    model=self.model,
                    inputs=batch
                )
                batch_embeddings = [item.embedding for item in batch_response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Failed to encode batch {i//batch_size}: {e}")
                for doc in batch:
                    try:
                        doc_response = client.embeddings.create(
                            model=self.model,
                            inputs=[doc]
                        )
                        all_embeddings.append(doc_response.data[0].embedding)
                    except Exception as doc_e:
                        logger.error(f"Failed to encode document: {doc_e}")
                        all_embeddings.append([0.0] * len(query_embedding))
        
        document_embeddings = np.array(all_embeddings)
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(document_embeddings, axis=1)
        
        similarity = np.dot(document_embeddings, query_embedding) / (doc_norms * query_norm + 1e-8)
        top_k_retrieval = min(top_k, len(documents))
        top_k_indices = np.argsort(similarity)[-top_k_retrieval:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append((int(idx), float(similarity[idx]), documents[idx]))
        
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

def run_retrieval_process(dataset, retriever, corpus_dir, method, batch_size, top_k):
    """
    Run retrieval for all instances with a single chunking method.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing with {method} chunking method")
    logger.info(f"{'='*60}\n")
    retrieved_docs = {}
    for instance in tqdm(dataset, desc=f"Retrieval with {method}"):
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

            results = retriever.retrieve(problem_statement, documents, batch_size, top_k)

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

def run_retrieval(model: str, dataset_name: str, split: str, corpus_dir: str, output_dir: str, chunking_method: str, top_k: int, device: str, batch_size: int):
    """
    Run retrieval for all instances in the dataset.
    """
    logger.info(f"Loading {dataset_name} {split} split...")
    dataset = load_dataset(dataset_name, split=split)

    retriever = EmbeddingRetriever(model_name=model, device=device)
    
    methods_to_process = []
    if chunking_method == "all":
        methods_to_process = ["sliding", "function", "hierarchical", "cAST", "natural"]
        # methods_to_process = ["function", "hierarchical", "cAST"]
    else:
        methods_to_process = [chunking_method]

    for method in methods_to_process:
        retrieved_docs = run_retrieval_process(
            dataset=dataset,
            retriever=retriever,
            corpus_dir=corpus_dir,
            method=method,
            batch_size=batch_size,
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
                        choices=["sliding", "function", "hierarchical", "cAST", "natural", "all"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="HuggingFace model id for embedding")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for embedding documents")
    args = parser.parse_args()
    
    run_retrieval(
        model=args.model,
        dataset_name=args.dataset,
        split=args.split,
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        chunking_method=args.method,
        top_k=args.top_k,
        device=args.device,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()