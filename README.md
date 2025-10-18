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