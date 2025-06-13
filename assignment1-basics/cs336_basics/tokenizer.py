from typing import Iterable
from re import escape
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer():
    def __init__(self, vocab:  dict[int, bytes], merges: list[tuple[bytes, bytes]],  special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else None

        if self.special_tokens:
            self.special_pattern = "|".join(f"({escape(token)})" for token in self.special_tokens)
            for token in self.special_tokens:
                token = token.encode('utf-8')
                # the repeated tokens should not be added to the vocab
                if token not in self.vocab.values():
                    self.vocab[len(self.vocab)] = token
        else:
            self.special_pattern = None

        self.pretoken_pattern = re.compile(PAT, flags=re.UNICODE)
        self.inverted_vocab = {v: k for k, v in self.vocab.items()}

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

    def _bpe_merge(self, pre_token: str) -> list[bytes]:
        # Convert pre_token to a list of bytes
        list_token = [bytes([char]) for char in pre_token.encode('utf-8')]
        
        for merge in self.merges:
            first, second = merge
            new_list_token = []
            i = 0
            while i < len(list_token):
                if i < len(list_token) - 1 and list_token[i] == first and list_token[i + 1] == second:
                    new_list_token.append(first + second)
                    i += 2
                else:
                    new_list_token.append(list_token[i])
                    i += 1
            list_token = new_list_token
        return list_token

    def encode(self, text: str) -> list[int]:
        special_tokens = self.special_tokens if self.special_tokens is not None else []

        if special_tokens and self.special_pattern:
            parts = re.split(self.special_pattern, text)
            parts = [part for part in parts if part]
        else:
            parts = [text]

        ids: list[int] = []
        for part in parts:
            if part in special_tokens:
                ids.append(self.inverted_vocab[part.encode('utf-8')])
                continue

            pre_tokens = re.findall(self.pretoken_pattern, part)

            for pre_token in pre_tokens:
                if pre_token.encode('utf-8') in self.vocab.values():
                    ids.append(self.inverted_vocab[pre_token.encode('utf-8')])
                    continue

                merged_token = self._bpe_merge(pre_token)
                for token in merged_token:
                    if self.inverted_vocab.get(token) is not None:
                        ids.append(self.inverted_vocab[token])
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        text = b"".join(self.vocab[id] for id in ids)
        return text.decode('utf-8', errors='replace')

