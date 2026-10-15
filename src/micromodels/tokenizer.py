"""
Custom BPE (Byte Pair Encoding) Tokenizer for Argent MicroModels.

Trained specifically on Argent developer queries (Russian, English, code paths, symbols, numbers).
Provides deterministic tokenization, padding, truncation, and vocabulary management without external deps.
"""

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from logger import get_logger

logger = get_logger("bpe_tokenizer")

class BPETokenizer:
    """Byte-Pair Encoding Tokenizer trained on Argent's CLI command distribution."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    CLS_TOKEN = "<cls>"
    SEP_TOKEN = "<sep>"
    MASK_TOKEN = "<mask>"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]

    def __init__(self, vocab_size: int = 4096):
        self.target_vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

        # Pre-populate special tokens
        for idx, token in enumerate(self.SPECIAL_TOKENS):
            self.vocab[token] = idx
            self.id2token[idx] = token

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.vocab[self.UNK_TOKEN]

    @property
    def cls_token_id(self) -> int:
        return self.vocab[self.CLS_TOKEN]

    @property
    def sep_token_id(self) -> int:
        return self.vocab[self.SEP_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _pre_tokenize(self, text: Any) -> List[str]:
        """Splits text into words/chunks while preserving punctuation, symbols, and whitespace."""
        if text is None:
            return []
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return []
        pattern = re.compile(
            r"""\s+|[A-Za-z]+|[А-Яа-яЁё]+|\d+|[^\s]""",
            re.UNICODE,
        )
        matches = pattern.findall(text)
        return [m for m in matches if m]

    def train(self, texts: List[str], min_frequency: int = 2) -> None:
        """Trains BPE merges and vocabulary on a list of texts."""
        logger.info(f"Training BPE tokenizer on {len(texts)} texts...")

        # 1. Collect all initial characters across the dataset
        char_counts = Counter()
        word_freqs = Counter()

        for text in texts:
            words = self._pre_tokenize(text)
            for w in words:
                word_freqs[w] += 1
                for ch in w:
                    char_counts[ch] += 1

        # Add initial characters to vocab
        next_id = len(self.vocab)
        for ch, _ in char_counts.most_common():
            if ch not in self.vocab:
                self.vocab[ch] = next_id
                self.id2token[next_id] = ch
                next_id += 1

        # Represent words as lists of symbols
        # e.g. "pytest" -> ["p", "y", "t", "e", "s", "t"]
        splits: Dict[str, List[str]] = {
            w: list(w) for w in word_freqs.keys()
        }

        # 2. Learn BPE merges
        num_merges = self.target_vocab_size - len(self.vocab)
        logger.info(f"Base characters: {len(self.vocab)}, Target merges: {num_merges}")

        for merge_idx in range(num_merges):
            pair_counts: Counter[Tuple[str, str]] = Counter()

            for word, freq in word_freqs.items():
                w_split = splits[word]
                for i in range(len(w_split) - 1):
                    pair = (w_split[i], w_split[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_frequency:
                # No more frequent pairs to merge
                break

            # Add merged token to vocabulary
            merged_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = merge_idx

            if merged_token not in self.vocab:
                self.vocab[merged_token] = next_id
                self.id2token[next_id] = merged_token
                next_id += 1

            # Update splits
            for word in list(splits.keys()):
                w_split = splits[word]
                i = 0
                new_split: List[str] = []
                while i < len(w_split):
                    if (
                        i < len(w_split) - 1
                        and w_split[i] == best_pair[0]
                        and w_split[i + 1] == best_pair[1]
                    ):
                        new_split.append(merged_token)
                        i += 2
                    else:
                        new_split.append(w_split[i])
                        i += 1
                splits[word] = new_split

        logger.info(
            f"BPE training complete. Total vocabulary size: {len(self.vocab)}, merges: {len(self.merges)}"
        )

    def _tokenize_word(self, word: str) -> List[str]:
        """Applies learned BPE merges to a single pre-tokenized word/chunk."""
        if not word:
            return []
        if word in self.vocab:
            return [word]

        symbols = list(word)
        if len(symbols) <= 1:
            return symbols

        while True:
            # Find all adjacent pairs
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            # Find the pair with the lowest merge rank
            best_pair = None
            min_rank = float("inf")
            for pair in pairs:
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    best_pair = pair

            if best_pair is None or min_rank == float("inf"):
                break

            # Merge occurrences of best_pair
            new_symbols = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == best_pair[0]
                    and symbols[i + 1] == best_pair[1]
                ):
                    new_symbols.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text into a list of BPE subword tokens."""
        words = self._pre_tokenize(text)
        tokens: List[str] = []
        for w in words:
            tokens.extend(self._tokenize_word(w))
        return tokens

    def encode(
        self,
        text: Any,
        add_special_tokens: bool = True,
        max_length: Optional[int] = 64,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> List[int]:
        """Encodes text to token IDs with optional special tokens and padding."""
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)

        tokens = self.tokenize(text)

        token_ids: List[int] = [
            self.vocab.get(tok, self.unk_token_id) for tok in tokens
        ]

        if add_special_tokens:
            token_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]

        if max_length is not None:
            if truncation and len(token_ids) > max_length:
                # Keep CLS at start and SEP at end if present
                if add_special_tokens and len(token_ids) >= 2:
                    token_ids = token_ids[: max_length - 1] + [self.sep_token_id]
                else:
                    token_ids = token_ids[:max_length]

            if padding == "max_length" and len(token_ids) < max_length:
                pad_count = max_length - len(token_ids)
                token_ids.extend([self.pad_token_id] * pad_count)

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs back to a text string."""
        tokens: List[str] = []
        for tid in token_ids:
            tok = self.id2token.get(tid, self.UNK_TOKEN)
            if skip_special_tokens and tok in self.SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return "".join(tokens)

    def save(self, file_path: Union[str, Path]) -> None:
        """Saves tokenizer vocabulary and merges to a JSON file atomically."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "target_vocab_size": self.target_vocab_size,
            "vocab": self.vocab,
            "merges": self.merges,
        }
        temp_file = file_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, file_path)
        logger.info(f"Tokenizer saved to {file_path}")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "BPETokenizer":
        """Loads tokenizer from a saved JSON file."""
        file_path = Path(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = cls(vocab_size=data.get("target_vocab_size", 4096))
        tok.vocab = data["vocab"]
        tok.id2token = {int(v): k for k, v in tok.vocab.items()}
        tok.merges = [tuple(m) for m in data.get("merges", [])]
        tok.merge_ranks = {tuple(m): idx for idx, m in enumerate(tok.merges)}
        return tok


if __name__ == "__main__":
    # Test tokenizer training on training pairs
    data_path = Path(__file__).parent / "data" / "training_pairs.jsonl"
    if data_path.exists():
        texts = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                texts.append(rec["text"])

        tokenizer = BPETokenizer(vocab_size=2048)
        tokenizer.train(texts)
        save_path = Path(__file__).parent / "data" / "tokenizer.json"
        tokenizer.save(save_path)

        # Test sample encoding
        sample_ru = "прочитай src/agent.py"
        sample_en = "run pytest tests/test_agent.py"
        encoded_ru = tokenizer.encode(sample_ru, max_length=16)
        encoded_en = tokenizer.encode(sample_en, max_length=16)
        print(f"Sample RU: '{sample_ru}' -> {encoded_ru}")
        print(f"Decoded RU: '{tokenizer.decode(encoded_ru)}'")
        print(f"Sample EN: '{sample_en}' -> {encoded_en}")
        print(f"Decoded EN: '{tokenizer.decode(encoded_en)}'")
