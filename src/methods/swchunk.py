from typing import List

class SlidingWindowChunk:
    def __init__(self, **configs):
        """
        A sliding window chunker that splits long code text into overlapping chunks.
        """
        self.max_chunk_size: int = configs['max_chunk_size']
        self.overlap_lines: int = configs['overlap_lines']

    @property
    def name(self) -> str:
        return "SlidingWindow"

    def chunkify(self, code: str, **kwargs) -> List[str]:
        """
        Split code text into overlapping chunks.
        
        Args:
            code (str): The code text to be chunked
            
        Returns:
            List[str]: List of code chunks, each containing at most max_chunk_size non-empty lines,
                      with overlap_lines overlapping lines between adjacent chunks
        """
        lines = code.split('\n')
        chunks = []
        cur = []

        for line in lines:
            if not line.strip():
                continue
            
            cur.append(line)
            
            if len(cur) >= self.max_chunk_size:
                chunks.append('\n'.join(cur))
                cur = cur[-self.overlap_lines:] if self.overlap_lines > 0 else []

        if cur:
            chunks.append('\n'.join(cur))

        return chunks
