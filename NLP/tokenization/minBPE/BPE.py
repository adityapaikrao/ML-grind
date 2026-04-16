"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.

"""
from typing import List, Tuple, Dict
from tqdm import tqdm

from .base import Tokenizer
from .utils.helper import get_pairs, merge


class BPETokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """
        Trains the BPETokenizer on the given text
        """
        assert vocab_size >= 256, "Vocab Size Error: Min Vocab size is 256 (for all byte values)"

        num_merges = vocab_size - 256

        raw_text = text.encode("utf-8") # get the raw bytes from the given text
        ids = list(map(int, raw_text)) # map bytes to int

        merges = {} # maps the (byte, byte) pair to the new idx: (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for itr in tqdm(range(num_merges), desc="Training BPE Tokenizer"):
            # get counts of the current byte pairs
            count_pairs = get_pairs(ids)

            # stop early if no pairs are left to merge
            if not count_pairs:
                break

            # get pair with highest count
            max_pair = max(count_pairs, key = count_pairs.get)

            # new idx for the pair
            new_idx = 256 + itr

            # replace the pair in the ids
            ids = merge(ids, max_pair, new_idx)

            # save the merge pair
            merges[max_pair] = new_idx
            vocab[new_idx] = vocab[max_pair[0]] + vocab[max_pair[1]]

            # (Optionally) print the merges
            if verbose: print(f"Merge Step {itr}: Merging pair {max_pair} with new id {new_idx}")
        
        self.vocab = vocab # used when we want to decode a token index stream
        self.merges = merges # used when we want to encode a string

        return

    def decode(self, ids: List[int]) -> str:
        """
        Decodes a given List of tokens (int) to a string
        """
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors = "replace")
        return text

    def encode(self, text: str) -> List[int]:
        """
        Encodes a given text to return a List of token ids
        """

        text_bytes = text.encode("utf-8")
        ids = list(map(int, text_bytes))

        while len(ids) >= 2:
            counts = get_pairs(ids)

            # find the pair with the least merge index to ensure causality in merges 
            # i.e ensure that we merge the pairs with index 259 before so that we can later merge pairs of (259, 213) etc.

            merges_pairs = {pair: self.merges.get(pair, float("inf")) for pair in counts.keys()} # pair -> token index
            min_merge_pair = min(merges_pairs, key=lambda p: merges_pairs[p])

            if min_merge_pair not in self.merges: 
                # new pair we havent seen in training before 
                # Since this was min merge pair => every other pair was also not seen => cant merge anymore
                break 
            
            new_idx = self.merges[min_merge_pair]
            ids = merge(ids, min_merge_pair, new_idx)

        return ids