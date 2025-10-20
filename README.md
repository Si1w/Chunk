# An Empirical Study on Chunk Strategies

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Si1w/Chunk.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Fetched Repositories

Run the script `eval/swebench/fetch_repos.py` to fetch repositories from the SWE-bench Lite dataset.

## Chunkify Repositories

```bash
python -m eval.swebench.chunkify_repos \ 
--method <chunking_method> \
--max_chunk_size <number of lines> \
--overlap_lines <number of lines>
```

## Generate Predictions

Run the script `eval/swebench/generate_predictions.py` to generate predictions for the chunked repositories.

## Evaluation Retrieval

```bash
python -m swebench.harness.run_evaluation \
-d SWE-bench/SWE-bench_Lite \
-s dev \
-p ./eval/swebench/predictions/codestral/Qwen3-Embedding-0.6B/<method> \
-id cast_dev
```