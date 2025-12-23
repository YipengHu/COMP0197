from dataclasses import dataclass
from typing import Dict, List, Sequence

@dataclass
class CharVocab:
    chars: List[str]            # index -> char
    stoi: Dict[str, int]        # char -> index

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

def build_char_vocab(text: str) -> CharVocab:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    return CharVocab(chars=chars, stoi=stoi)

def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi[c] for c in text]

def decode(ids: Sequence[int], chars: Sequence[str]) -> str:
    return "".join(chars[int(i)] for i in ids)
