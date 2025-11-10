from typing import List, Dict, Any

import tree_sitter as ts
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_c_sharp as tscsharp
import tree_sitter_typescript as tstypescript

from .utils import build_metadata, extract_function_name, to_code_window

class FunctionLevelChunk:
    
    def __init__(self, **configs):
        self.max_chunk_size: int = configs['max_chunk_size']
        self.language: str = configs['language']
        self.metadata_template: str = configs['metadata_template']
        self.overlap_lines: int = configs['overlap_lines']

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
        if repo_level_metadata is None:
            repo_level_metadata = {}
            
        tree = self.parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node

        def extract_functions_recursively(node):
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
            line_count = len(func_code.splitlines())
            func_name = extract_function_name(node)
            start_line = node.start_point[0] + 1
            
            if line_count <= self.max_chunk_size:
                chunk_metadata = build_metadata(
                    metadata_template=self.metadata_template,
                    func_name=func_name,
                    node_type=node.type,
                    start_line=start_line,
                    end_line=node.end_point[0] + 1,
                    lines=line_count,
                    max_chunk_size=self.max_chunk_size,
                    repo_level_metadata=repo_level_metadata
                )
                
                chunk_dict = to_code_window(self.metadata_template, func_code, chunk_metadata)
                chunks.append(chunk_dict)
            else:
                lines = func_code.splitlines(keepends=True)
                chunk_index = 0
                i = 0
                
                while i < len(lines):
                    end_idx = min(i + self.max_chunk_size, len(lines))
                    chunk_lines = lines[i:end_idx]
                    chunk_code = ''.join(chunk_lines)
                    
                    chunk_metadata = build_metadata(
                        metadata_template=self.metadata_template,
                        func_name=func_name,
                        node_type=node.type,
                        start_line=start_line + i,
                        end_line=start_line + end_idx - 1,
                        lines=len(chunk_lines),
                        max_chunk_size=self.max_chunk_size,
                        repo_level_metadata=repo_level_metadata
                    )
                    
                    chunk_dict = to_code_window(self.metadata_template, chunk_code, chunk_metadata)
                    chunks.append(chunk_dict)
                    
                    if end_idx < len(lines):
                        i += self.max_chunk_size - self.overlap_lines
                    else:
                        break
                        
                    chunk_index += 1

        return chunks