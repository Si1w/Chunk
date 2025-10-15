from typing import Any, Dict
import numpy as np
import string

def preprocess_nws_count(bstring: bytes) -> np.ndarray:
    whitespace_bytes = tuple(ord(x) for x in string.whitespace)
    is_nws = np.array([x not in whitespace_bytes for x in bstring])
    is_nws_cumsum = np.cumsum(is_nws)
    nws_cumsum = np.concatenate([[0], is_nws_cumsum])
    return nws_cumsum

def build_metadata(metadata_template: str, func_name: str, node_type: str, start_line: int, end_line: int, 
                    max_chunk_size: int, lines: int, repo_level_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build metadata for the chunk.

    Args:
        func_name: Function name
        node_type: AST node type
        start_line: Starting line number
        end_line: Ending line number  
        lines: Number of lines
        repo_level_metadata: Repository-level metadata
    """
    if metadata_template == "none":
        return {}
    elif metadata_template == "default":
        filepath = repo_level_metadata.get("filepath", "")
        return {
            "filepath": filepath,
            "function_name": func_name,
            "type": node_type,
            "chunk_size": max_chunk_size,
            "line_count": lines,
            "start_line_no": start_line,
            "end_line_no": end_line,
        }
    elif metadata_template == "coderagbench-repoeval":
        fpath_tuple = repo_level_metadata.get("fpath_tuple", [])
        repo = repo_level_metadata.get("repo", "")
        return {
            "fpath_tuple": fpath_tuple,
            "repo": repo,
            "function_name": func_name,
            "type": node_type,
            "chunk_size": max_chunk_size,
            "line_count": lines,
            "start_line_no": start_line,
            "end_line_no": end_line,
        }
    elif metadata_template == "coderagbench-swebench-lite":
        instance_id = repo_level_metadata.get("instance_id", "")
        filename = repo_level_metadata.get("filename", "")
        return {
            "_id": f"{instance_id}_{func_name}_{start_line}-{end_line}",
            "title": filename,
            "function_name": func_name,
            "instance_id": instance_id,
        }
    else:
        raise ValueError(f"Unsupported Metadata Template Name: {metadata_template}!")
    
def extract_function_name(node) -> str:
    """Extract function name from tree-sitter node."""
    for child in node.children:
        if child.type == 'identifier':
            return child.text.decode('utf-8')
    return "unknown_function"

def to_code_window(metadata_template: str, chunk_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the chunk into a code window for downstream integration.
    """
    if metadata_template == "coderagbench-swebench-lite":
        return {
            "_id": metadata["_id"],
            "title": metadata['title'],
            "text": chunk_text
        }
    else:
        return {
            "content": chunk_text,
            "metadata": metadata
        }