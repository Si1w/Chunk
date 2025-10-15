from typing import List, Dict, Any

import tree_sitter as ts
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_c_sharp as tscsharp
import tree_sitter_typescript as tstypescript

from .utils import build_metadata, extract_function_name, to_code_window

class FunctionLevelChunk:
    """
    A function-level chunker that extracts individual functions/methods from code.
    
    Uses tree-sitter parsers to identify and extract function definitions
    based on the programming language syntax.
    """
    
    def __init__(self, **configs):
        self.max_chunk_size: int = configs['max_chunk_size']
        self.language: str = configs['language']
        self.metadata_template: str = configs['metadata_template']

        if self.language == "python":
            self.parser = ts.Parser(ts.Language(tspython.language()))
        elif self.language == "java":
            self.parser = ts.Parser(ts.Language(tsjava.language()))
        elif self.language == "csharp":
            self.parser = ts.Parser(ts.Language(tscsharp.language()))
        elif self.language == "typescript":
            self.parser = ts.Parser(ts.Language(tstypescript.language_tsx()))
        else:
            raise ValueError(f"Unsupported Programming Language: {self.language}!")
        
    @property
    def name(self) -> str:
        return "FunctionLevel"

    def chunkify(self, code: str, repo_level_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split code text into function-level chunks.
        
        Args:
            code (str): The code text to be chunked
            repo_level_metadata (Dict[str, Any]): Repository-level metadata
            
        Returns:
            List[Dict[str, Any]]: List of chunk dictionaries with astchunk-compatible output format
        """
        if repo_level_metadata is None:
            repo_level_metadata = {}
            
        tree = self.parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node

        def extract_functions_recursively(node):
            """Recursively extract all function/method nodes from the AST tree."""
            functions = []
            
            if node.type in ['function_definition', 'method_declaration', 'function_declaration']:
                functions.append(node)
            
            for child in node.children:
                functions.extend(extract_functions_recursively(child))
                
            return functions

        chunks = []
        function_nodes = extract_functions_recursively(root_node)
        
        for node in function_nodes:
            start_byte = node.start_byte
            end_byte = node.end_byte
            func_code = code[start_byte:end_byte]
            
            if len(func_code.splitlines()) <= self.max_chunk_size:
                func_name = extract_function_name(node)
                chunk_metadata = build_metadata(
                    metadata_template=self.metadata_template,
                    func_name=func_name,
                    node_type=node.type,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    lines=len(func_code.splitlines()),
                    max_chunk_size=self.max_chunk_size,
                    repo_level_metadata=repo_level_metadata
                )
                
                chunk_dict = to_code_window(self.metadata_template, func_code, chunk_metadata)
                chunks.append(chunk_dict)

        return chunks