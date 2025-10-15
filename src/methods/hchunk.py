from typing import List, Dict, Any

import tree_sitter as ts
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_c_sharp as tscsharp
import tree_sitter_typescript as tstypescript
import builtins

from .utils import extract_function_name, to_code_window

class HierarchicalChunk:
    def __init__(self, **configs):
        self.max_chunk_size: int = configs['max_chunk_size']
        self.language: str = configs['language']
        self.metadata_template: str = configs['metadata_template']

        if self.language == "python":
            self.parser = ts.Parser(ts.Language(tspython.language()))
            self.builtin_functions = set(dir(builtins))
        elif self.language == "java":
            self.parser = ts.Parser(ts.Language(tsjava.language()))
            self.builtin_functions = {
                'System', 'String', 'Integer', 'Double', 'Boolean', 'Math', 'Objects', 'Arrays',
                'Collections', 'List', 'Map', 'Set', 'ArrayList', 'HashMap', 'HashSet'
            }
        elif self.language == "csharp":
            self.parser = ts.Parser(ts.Language(tscsharp.language()))
            self.builtin_functions = {
                'Console', 'Convert', 'Math', 'Object', 'String', 'Int32', 'Double', 'Boolean',
                'List', 'Dictionary', 'Array', 'Enum', 'DateTime', 'Guid'
            }
        elif self.language == "typescript":
            self.parser = ts.Parser(ts.Language(tstypescript.language_tsx()))
            self.builtin_functions = {
                'console', 'parseInt', 'parseFloat', 'isNaN', 'isFinite', 'encodeURI', 'decodeURI',
                'String', 'Number', 'Boolean', 'Array', 'Object', 'Math', 'Date', 'RegExp',
                'JSON', 'Promise', 'Set', 'Map', 'WeakSet', 'WeakMap'
            }
        else:
            raise ValueError(f"Unsupported Programming Language: {self.language}!")
    
    @property
    def name(self) -> str:
        return "Hierarchical"
        
    def chunkify(self, code: str, repo_level_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split code text into hierarchical chunks with class and function levels.
        
        Args:
            code (str): The code text to be chunked
            repo_level_metadata (Dict[str, Any]): Repository-level metadata
            
        Returns:
            List[Dict[str, Any]]: List of chunk dictionaries with hierarchical relationships
        """
        if repo_level_metadata is None:
            repo_level_metadata = {}
            
        tree = self.parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node

        def count_child_nodes(node) -> int:
            """Count child nodes recursively."""
            count = 1
            for child in node.children:
                count += count_child_nodes(child)
            return count

        def extract_inheritance_info(node) -> List[str]:
            """Extract inheritance/implementation information from class node."""
            inherited_classes = []
            
            if self.language == "python":
                for child in node.children:
                    if child.type == 'argument_list':
                        for arg in child.children:
                            if arg.type == 'identifier':
                                inherited_classes.append(arg.text.decode('utf-8'))
                            elif arg.type == 'attribute':
                                inherited_classes.append(arg.text.decode('utf-8'))
                                
            elif self.language == "java":
                for child in node.children:
                    if child.type == 'superclass':
                        for grandchild in child.children:
                            if grandchild.type == 'type_identifier':
                                inherited_classes.append(grandchild.text.decode('utf-8'))
                    elif child.type == 'super_interfaces':
                        for grandchild in child.children:
                            if grandchild.type == 'type_identifier':
                                inherited_classes.append(grandchild.text.decode('utf-8'))
                                
            elif self.language == "csharp":
                for child in node.children:
                    if child.type == 'base_list':
                        for grandchild in child.children:
                            if grandchild.type == 'identifier':
                                inherited_classes.append(grandchild.text.decode('utf-8'))
                                    
            elif self.language == "typescript":
                for child in node.children:
                    if child.type == 'class_heritage':
                        for grandchild in child.children:
                            if grandchild.type == 'extends_clause':
                                for ggchild in grandchild.children:
                                    if ggchild.type == 'identifier':
                                        inherited_classes.append(ggchild.text.decode('utf-8'))
                            elif grandchild.type == 'implements_clause':
                                for ggchild in grandchild.children:
                                    if ggchild.type == 'identifier':
                                        inherited_classes.append(ggchild.text.decode('utf-8'))
                
            return inherited_classes

        def find_function_calls(node) -> List[str]:
            """Find all user-defined function calls within a node."""
            calls = []
            
            def extract_calls(n):
                """Extract function calls from AST node."""
                if n.type == 'call':
                    for child in n.children:
                        if child.type == 'identifier':
                            func_name = child.text.decode('utf-8')
                            if func_name not in self.builtin_functions and func_name not in calls:
                                calls.append(func_name)
                        elif child.type == 'attribute':
                            attr_name = child.text.decode('utf-8')
                            if attr_name not in self.builtin_functions and attr_name not in calls:
                                calls.append(attr_name)
                
                for child in n.children:
                    extract_calls(child)
            
            extract_calls(node)
            return calls

        def build_relationship_dict(node) -> Dict[str, List[str]]:
            """Build relationship dictionary for a node."""
            relationship = {}
            
            if node.type == 'class_definition':
                inherited_classes = extract_inheritance_info(node)
                if inherited_classes:
                    relationship['inherit'] = inherited_classes
                    
            elif node.type == 'function_definition':
                function_calls = find_function_calls(node)
                if function_calls:
                    relationship['call'] = function_calls
            
            return relationship if relationship else None

        def process_class(node, code: str) -> List[Dict[str, Any]]:
            """Process a class node, combining class definition with private methods."""
            start_byte = node.start_byte
            end_byte = node.end_byte
            class_code = code[start_byte:end_byte]
            
            class_name = extract_function_name(node)
            relationship = build_relationship_dict(node)
            
            private_methods = []
            public_methods = []
            
            def find_methods_in_class(class_node):
                """Find public and private methods in class."""
                for child in class_node.children:
                    if child.type == 'block' or child.type == 'class_body':
                        for grandchild in child.children:
                            if grandchild.type == 'function_definition':
                                method_name = extract_function_name(grandchild)
                                if method_name and method_name.startswith('_'):
                                    private_methods.append(grandchild)
                                else:
                                    public_methods.append(grandchild)
                    elif child.type == 'function_definition':
                        method_name = extract_function_name(child)
                        if method_name and method_name.startswith('_'):
                            private_methods.append(child)
                        else:
                            public_methods.append(child)
            
            find_methods_in_class(node)
            
            if private_methods:
                class_lines = class_code.splitlines()
                filtered_lines = []
                public_method_ranges = []
                
                for pub_method in public_methods:
                    start_line = pub_method.start_point[0] - node.start_point[0]
                    end_line = pub_method.end_point[0] - node.start_point[0]
                    public_method_ranges.append((start_line, end_line))
                
                public_method_ranges.sort()
                
                last_end = 0
                for start, end in public_method_ranges:
                    filtered_lines.extend(class_lines[last_end:start])
                    last_end = end + 1
                filtered_lines.extend(class_lines[last_end:])
                
                filtered_class_code = '\n'.join(filtered_lines)
            else:
                filtered_class_code = class_code
            
            chunk_metadata = {
                'class_name': class_name,
                'chunk_type': 'class',
                'relationship': relationship,
                'parent_name': None,
                'private_methods_count': len(private_methods),
                'public_methods_count': len(public_methods),
                'node_count': count_child_nodes(node),
                'chunk_size': len(filtered_class_code.encode('utf-8'))
            }
            
            class_chunk = to_code_window(self.metadata_template, filtered_class_code, chunk_metadata)
            chunks_list = [class_chunk]
            
            for pub_method in public_methods:
                method_start = pub_method.start_byte
                method_end = pub_method.end_byte
                method_code = code[method_start:method_end]
                method_name = extract_function_name(pub_method)
                method_relationship = build_relationship_dict(pub_method)
                
                method_metadata = {
                    'func_name': method_name,
                    'chunk_type': 'function',
                    'relationship': method_relationship,
                    'class_name': class_name
                }
                
                method_chunk = to_code_window(self.metadata_template, method_code, method_metadata)
                chunks_list.append(method_chunk)
            
            return chunks_list

        def process_function(node, code: str, parent_name: str = None) -> Dict[str, Any]:
            """Process a function node into a chunk."""
            start_byte = node.start_byte
            end_byte = node.end_byte
            func_code = code[start_byte:end_byte]
            
            func_name = extract_function_name(node)
            relationship = build_relationship_dict(node)
            
            chunk_metadata = {
                'func_name': func_name,
                'chunk_type': 'function',
                'relationship': relationship,
                'class_name': parent_name
            }
            
            return to_code_window(self.metadata_template, func_code, chunk_metadata)

        def traverse_node_recursively(node, parent_name: str = None):
            """Recursively traverse AST nodes to process classes and functions."""
            if node.type == 'class_definition':
                class_chunks = process_class(node, code)
                if class_chunks:
                    chunks.extend(class_chunks)
                    
            elif node.type == 'function_definition' and parent_name is None:
                func_chunk = process_function(node, code, parent_name)
                if func_chunk:
                    chunks.append(func_chunk)
            else:
                for child in node.children:
                    traverse_node_recursively(child, parent_name)

        chunks = []
        traverse_node_recursively(root_node)

        return chunks