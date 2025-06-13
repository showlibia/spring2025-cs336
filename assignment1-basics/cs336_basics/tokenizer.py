from typing import Iterable
from re import escape
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer():
    def __init__(self, vocab:  dict[int, bytes], merges: list[tuple[bytes, bytes]],  special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.merge_ranks = {merge: i for i, merge in enumerate(merges)}
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else None

        if self.special_tokens:
            self.special_pattern = "|".join(f"({escape(token)})" for token in self.special_tokens)
            self.special_pattern = re.compile(self.special_pattern, flags=re.UNICODE)
            
            for token in self.special_tokens:
                token = token.encode('utf-8')
                # the repeated tokens should not be added to the vocab
                if token not in self.vocab.values():
                    self.vocab[len(self.vocab)] = token
        else:
            self.special_pattern = None

        self.pretoken_pattern = re.compile(PAT, flags=re.UNICODE)
        self.inverted_vocab = {v: k for k, v in self.vocab.items()}
        self.encode_cache = {}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        import json
        with open(vocab_filepath, 'rb') as f:
            vocab = json.load(f)
        if not isinstance(vocab, dict):
            raise ValueError("Vocabulary file must contain a JSON object mapping integers to byte strings.")
        vocab = {int(k): v.encode('utf-8') for k, v in vocab.items()}
        if special_tokens is not None:
            vocab.update({len(vocab) + i: token.encode('utf-8') for i, token in enumerate(special_tokens)})

        with open(merges_filepath, 'rb') as f:
            merges = [tuple(line.strip().split()) for line in f.readlines()]
            merges = [(first, second) for first, second in merges]

        return cls(vocab, merges, special_tokens)

    def _get_pairs(self, list_token: list[bytes]) -> set[tuple[bytes, bytes]]:
        return set((list_token[i], list_token[i+1]) for i in range(len(list_token) - 1))

    def _bpe_merge(self, pre_token: str) -> list[bytes]:
        list_token = [bytes([b]) for b in pre_token.encode('utf-8')]
        
        while True:
            pairs = self._get_pairs(list_token)
            if not pairs:
                break
            # Find the best pair to merge, i.e., the one with the lowest rank
            best_pair = min(pairs, key=lambda p: self.merge_ranks.get(p, float('inf')))
            if best_pair not in self.merge_ranks:
                break

            first, second = best_pair
            new_list_token = []
            i = 0
            while i < len(list_token):
                if i < len(list_token) - 1 and list_token[i] == first and list_token[i+1] == second:
                    new_list_token.append(first + second)
                    i += 2
                else:
                    new_list_token.append(list_token[i])
                    i += 1
            list_token = new_list_token
            
        return list_token

    def encode(self, text: str) -> list[int]:
        special_tokens = self.special_tokens if self.special_tokens is not None else []
        
        if not special_tokens or not self.special_pattern:
            return self._encode(text)
        
        ids: list[int] = []
        current_pos = 0
        
        for match in re.finditer(self.special_pattern, text):
            if match.start() > current_pos:
                normal_text = text[current_pos:match.start()]
                ids.extend(self._encode(normal_text))

            special_token = match.group()
            if special_token in special_tokens:
                ids.append(self.inverted_vocab[special_token.encode('utf-8')])
            
            current_pos = match.end()
        
        if current_pos < len(text):
            remaining_text = text[current_pos:]
            ids.extend(self._encode(remaining_text))
        
        return ids

    def _encode(self, text: str) -> list[int]:
        ids: list[int] = []
        
        for match in re.finditer(self.pretoken_pattern, text):
            pre_token = match.group(0)
            pre_token_bytes = pre_token.encode('utf-8')
            
            if pre_token_bytes in self.inverted_vocab:
                ids.append(self.inverted_vocab[pre_token_bytes])
                continue

            if pre_token_bytes in self.encode_cache:
                ids.extend(self.encode_cache[pre_token_bytes])
                continue

            merged_token = self._bpe_merge(pre_token)
            token_ids = []
            for token in merged_token:
                if token in self.inverted_vocab:
                    token_ids.append(self.inverted_vocab[token])
            self.encode_cache[pre_token_bytes] = token_ids
            ids.extend(token_ids)
        
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        text = b"".join(self.vocab[id] for id in ids)
        return text.decode('utf-8', errors='replace')

