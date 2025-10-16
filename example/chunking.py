from src.methods import SlidingWindowChunk, FunctionLevelChunk, HierarchicalChunk
from tqdm import tqdm

def main():
    input_path = "example/example_repo.txt"

    with open(input_path, "r") as f:
        code=f.read()

    configs = {
        "max_chunk_size": 100,
        "language": "python",
        "metadata_template": "default",
        "chunk_expansion": False,
        "overlap_lines": 20
    }

    for method in [SlidingWindowChunk, FunctionLevelChunk, HierarchicalChunk]:
        chunker = method(**configs)
        chunks = chunker.chunkify(code)
    
        output_path = f"example/output/{chunker.name}_chunks.txt"

        if method == SlidingWindowChunk:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Chunking Method: {chunker.name}\n\n")
                
                for i, chunk in tqdm(enumerate(chunks, 1), total=len(chunks)):
                    line_count = len(chunk.split('\n'))
                    header = f"{'-' * 25} Chunk {i} ({line_count} lines) {'-' * 25}\n"
                    f.write(header)
                    f.write(chunk)
                    f.write("\n" + "-" * (len(header) - 1) + "\n\n")
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Chunking Method: {chunker.name}\n\n")

                for i, chunk in tqdm(enumerate(chunks, 1), total=len(chunks)):
                    content = chunk.get('content', chunk.get('context', ''))
                    metadata = chunk.get('metadata', {})
                    line_count = len(content.split('\n'))
                    header = f"{'-' * 25} Chunk {i} ({line_count} lines) {'-' * 25}\n"
                    f.write(header)
                    f.write(content)
                    f.write("\nMetadata:\n")
                    for key, value in metadata.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n" + "-" * (len(header) - 1) + "\n\n")

if __name__ == "__main__":
    main()