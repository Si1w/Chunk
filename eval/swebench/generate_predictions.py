import os
import re
import json
import time
import argparse
from getpass import getpass
from mistralai import Mistral
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

api_key = getpass("Enter your MISTRAL_API_KEY: ").strip()
os.environ["MISTRAL_API_KEY"] = api_key
client = Mistral(api_key=api_key.strip())

PROMPT_TEMPLATE = """
You will be provided with a partial code base and an issue statement explaining a problem to resolve.
<issue>
{problem_statement}
</issue>
<code>
{code_base}
</code>
I need you to solve this issue by generating a single patch file that I can apply directly to this repository using git apply. Please respond with a single patch file in the following format.
<patch>
--- a/file.py
+++ b/file.py
@@ -1,27 +1,35 @@
def euclidean(a, b):
- while b:
- a, b = b, a % b
- return a
+ if b == 0:
+ return a
+ return euclidean(b, a % b)


def bresenham(x0, y0, x1, y1):
points = []
dx = abs(x1 - x0)
dy = abs(y1 - y0)
- sx = 1 if x0 < x1 else -1
- sy = 1 if y0 < y1 else -1
- err = dx - dy
+ x, y = x0, y0
+ sx = -1 if x0 > x1 else 1
+ sy = -1 if y0 > y1 else 1

- while True:
- points.append((x0, y0))
- if x0 == x1 and y0 == y1:
- break
- e2 = 2 * err
- if e2 > -dy:
+ if dx > dy:
+ err = dx / 2.0
+ while x != x1:
+ points.append((x, y))
err -= dy
- x0 += sx
- if e2 < dx:
- err += dx
- y0 += sy
+ if err < 0:
+ y += sy
+ err += dx
+ x += sx
+ else:
+ err = dy / 2.0
+ while y != y1:
+ points.append((x, y))
+ err -= dx
+ if err < 0:
+ x += sx
+ err += dy
+ y += sy

+ points.append((x, y))
return points
</patch>
"""

def load_retrieved_doc(code_path: str) -> str:
    with open(code_path, 'r', encoding='utf-8') as d:
        return json.load(d)
    
def parse_patch(patch_str: str) -> str:
    """
    Parse and extract content from markdown code blocks (``` ```).
    
    Args:
        patch_str: String that may contain markdown code blocks
        
    Returns:
        Extracted content from code blocks, or original string if no code blocks found
    """
    pattern = r'```(?:\w+)?\n?(.*?)```'
    matches = re.findall(pattern, patch_str, re.DOTALL)
    
    if matches:
        return '\n\n'.join(match.strip() for match in matches)
    
    return patch_str.strip()

def llm_response(code_base: str, problem_statement: str) -> str:
    prompt = PROMPT_TEMPLATE.format(problem_statement=problem_statement, code_base=code_base)
    
    response = client.chat.complete(
        model="devstral-medium-latest",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=2048,
        temperature=0,
        random_seed=42,
    )
    
    return response.choices[0].message.content

def format_prediction(instance_id: str,prediction: str):
    pred = {
        "instance_id": instance_id,
        "model_name_or_path": "devstral-medium-latest",
        "model_patch": parse_patch(prediction)
    }

    return pred

def pred_generation(docs_dir: str, method: str, top_k: int):
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="dev")
    preds = {}
    for instance in tqdm(dataset, desc=f"Generating Predictions with {method}"):
        if instance["repo"] == "pvlib/pvlib-python" or instance["repo"] == "pydicom/pydicom":
            continue
        instance_id = instance["instance_id"]
        problem_statement = instance["problem_statement"]
        docs_path = Path(docs_dir, f"{method}_retrieval.json")
        docs = load_retrieved_doc(str(docs_path))
        doc = docs[instance_id]
        code_base = ""
        for i in range(top_k):
            code_base += doc["results"][i]["file_path"] + "\n"
            code_base += doc["results"][i]["content"] + "\n"
            metadata = doc["results"][i]["metadata"]
            code_base += str(metadata.get("relationship", "")) + "\n" if metadata.get("relationship") else ""
            code_base += str(metadata.get("parent_name", "")) + "\n" if metadata.get("parent_name") else ""
        prediction = llm_response(code_base, problem_statement)
        preds[instance_id] = format_prediction(instance_id, prediction)
        time.sleep(5) # To avoid rate limiting, adjust as needed
    return preds
        
def process_pred(method: str, model: str, top_k: int, docs_dir: str, output_dir: str):
    model_name = model.split("/")[-1]
    docs_path = f"{docs_dir}/{model_name}"
    output_path = f"{output_dir}/{model_name}"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    if method == "all":
        methods = ["sliding", "function", "hierarchical", "cAST"]
    else:
        methods = [method]
    
    for method in methods:
        logger.info(f"Generating predictions for method: {method}")
        preds = pred_generation(docs_path, method, top_k)
        output_path = f"{output_dir}/{model_name}/{method}_predictions.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(preds, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved predictions to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Predictions using Mistral")
    parser.add_argument("--method", type=str, default="all",
                        choices=["sliding", "function", "hierarchical", "cAST", "all"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="HuggingFace model id for embedding")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top retrieved documents to use as context")
    parser.add_argument("--docs-dir", type=str, default="./eval/swebench/retrieval",)
    parser.add_argument("--output-dir", type=str, default="./eval/swebench/predictions/codestral",
                        help="Directory to save predictions")
    args = parser.parse_args()

    process_pred(
        method=args.method,
        model=args.model,
        top_k=args.top_k,
        docs_dir=args.docs_dir,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()