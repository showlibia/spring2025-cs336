from doctest import Example
from email import errors
from email.policy import default
import os
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
import regex as re
from collections import defaultdict

example = """low low low low low lower lower widest widest widest newest newest newest newest newest newest"""

def bpe_example(chunk: str):
    matches = re.finditer(PAT, chunk)
    frequency_table: dict[tuple[bytes, ...], int] = defaultdict(int)
    for match in matches:
        word_tuple: tuple[bytes,...] = tuple(char.encode('utf-8') for char in match.group())

        frequency_table[word_tuple] += 1
    max_length = max(len(word) for word in frequency_table.keys())
    vocab: dict[int, bytes] = {}
    for id in range(max_length):
        pair_bytes:dict[str, int] = defaultdict(int)
        for word, frequency in frequency_table.items():
            for i in range(len(word) - 1):
                pair = bytes(word[i]).decode("utf-8") + bytes(word[i+1]).decode("utf-8")
                pair_bytes[pair] += frequency # type: ignore
        if not pair_bytes:
            break
        max_pair, _ = max(pair_bytes.items(), key=lambda item:(item[1], item[0]))
        vocab[id] = max_pair.encode("utf-8")
        new_frequency_table: dict[tuple[bytes, ...], int] = defaultdict(int)
        for word, freq in frequency_table.items():
            word_list = list(word)
            i = 0
            new_word = []
            while i < len(word_list):
                if  i < len(word_list) - 1 and bytes(word[i]).decode("utf-8") + bytes(word[i+1]).decode("utf-8") == max_pair:
                    merged = word_list[i] + word_list[i + 1]
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word_list[i])
                    i += 1
            new_frequency_table[tuple(new_word)] += freq
        frequency_table = new_frequency_table
## Usage
# with open("../data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
#     boundaries = find_chunk_boundaries(
#         f, 20, "<|endoftext|>".encode("utf-8"))
        
#     # The following is a serial implementation, but you can parallelize this 
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
# #         # Run pre-tokenization on your chunk and store the counts for each pre-token