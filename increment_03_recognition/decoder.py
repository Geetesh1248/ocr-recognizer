"""CTC decoding algorithms: greedy argmax and prefix beam search with character n-gram LM."""
from collections import defaultdict
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


def greedy_ctc(probabilities: np.ndarray, alphabet: str) -> Tuple[str, float]:
    """Collapse consecutive token repeats first, then remove token 0 (blank)."""
    ids = probabilities.argmax(axis=-1)
    text, scores, previous = [], [], None
    for step, token in enumerate(ids):
        token = int(token)
        if token != previous and token != 0:
            text.append(alphabet[token-1])
            scores.append(max(1e-12, float(probabilities[step, token])))
        previous = token
    confidence = math.exp(sum(math.log(s) for s in scores)/len(scores)) if scores else 0.0
    return "".join(text), confidence


class CharNgramLM:
    """Character n-gram language model with Laplace smoothing."""

    def __init__(self, alphabet: str, n: int = 4, alpha: float = 0.1):
        self.alphabet = alphabet
        self.vocab = set(alphabet)
        self.vocab_size = max(1, len(self.vocab))
        self.n = max(2, n)
        self.alpha = alpha
        self.ngram_counts: Dict[str, int] = defaultdict(int)
        self.context_counts: Dict[str, int] = defaultdict(int)

    def train_text(self, text: str):
        """Train language model on a text string, recording multi-order n-grams."""
        clean = "".join(ch for ch in text if ch in self.vocab)
        L = len(clean)
        for k in range(1, self.n + 1):
            for i in range(L - k + 1):
                gram = clean[i:i+k]
                self.ngram_counts[gram] += 1
                if k > 1:
                    ctx = gram[:-1]
                    self.context_counts[ctx] += 1

    def score(self, prefix: str, next_char: str) -> float:
        """Return log probability log P(next_char | prefix) using backoff to available context."""
        if next_char not in self.vocab:
            return -10.0
        for ctx_len in range(min(len(prefix), self.n - 1), -1, -1):
            context = prefix[-ctx_len:] if ctx_len > 0 else ""
            ngram = context + next_char
            c_ctx = self.context_counts.get(context, 0) if context else sum(self.ngram_counts.get(c, 0) for c in self.vocab)
            if c_ctx > 0 or ctx_len == 0:
                c_ngram = self.ngram_counts.get(ngram, 0)
                prob = (c_ngram + self.alpha) / (max(1, c_ctx) + self.alpha * self.vocab_size)
                return math.log(max(1e-12, prob))
        return math.log(1.0 / self.vocab_size)

    @classmethod
    def from_corpus(cls, corpus_dir: Path, alphabet: str, n: int = 4) -> "CharNgramLM":
        """Load and train LM on all .txt and .json files in corpus directory."""
        lm = cls(alphabet, n=n)
        corpus_dir = Path(corpus_dir)
        if not corpus_dir.is_dir():
            return lm
        for txt_file in sorted(corpus_dir.glob("*.txt")):
            try:
                lm.train_text(txt_file.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
        return lm


def beam_search_ctc(
    probabilities: np.ndarray,
    alphabet: str,
    beam_width: int = 10,
    lm_weight: float = 0.0,
    lm: Optional[CharNgramLM] = None
) -> Tuple[str, float]:
    """Prefix beam search decoding for CTC acoustic probabilities.
    
    Args:
        probabilities: (T, classes) matrix of softmax probabilities (blank is index 0).
        alphabet: String mapping index 1..C to characters.
        beam_width: Beam width (number of candidates preserved per step).
        lm_weight: Weight for language model log-probability bonus.
        lm: Optional CharNgramLM instance.
        
    Returns:
        (best_text, confidence)
    """
    if beam_width <= 1:
        return greedy_ctc(probabilities, alphabet)

    T, classes = probabilities.shape
    # Beam entry: prefix -> (p_blank, p_non_blank)
    beam: Dict[str, Tuple[float, float]] = {"": (1.0, 0.0)}

    for t in range(T):
        p_blank_t = max(1e-12, float(probabilities[t, 0]))
        next_beam: Dict[str, Tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))

        for prefix, (p_b, p_nb) in beam.items():
            p_total = p_b + p_nb
            if p_total <= 0:
                continue

            # 1. Blank transition
            nb_b, nb_nb = next_beam[prefix]
            next_beam[prefix] = (nb_b + p_total * p_blank_t, nb_nb)

            # 2. Non-blank transitions
            for c_idx in range(1, classes):
                p_c = max(1e-12, float(probabilities[t, c_idx]))
                char = alphabet[c_idx - 1]

                if prefix and prefix[-1] == char:
                    # Same character repeated:
                    # Transition from blank creates a new character
                    new_p = prefix + char
                    nb_b, nb_nb = next_beam[new_p]
                    next_beam[new_p] = (nb_b, nb_nb + p_b * p_c)

                    # Transition from non-blank collapses into existing prefix
                    nb_b, nb_nb = next_beam[prefix]
                    next_beam[prefix] = (nb_b, nb_nb + p_nb * p_c)
                else:
                    new_p = prefix + char
                    nb_b, nb_nb = next_beam[new_p]
                    next_beam[new_p] = (nb_b, nb_nb + p_total * p_c)

        # Prune to beam_width candidates based on score
        def get_score(item):
            pref, (pb, pnb) = item
            total_prob = pb + pnb
            if total_prob <= 0:
                return -1e18
            score = math.log(total_prob)
            if lm is not None and lm_weight > 0.0 and len(pref) > 0:
                score += lm_weight * lm.score(pref[:-1], pref[-1])
            return score

        sorted_items = sorted(next_beam.items(), key=get_score, reverse=True)[:beam_width]
        beam = dict(sorted_items)

    if not beam:
        return "", 0.0

    best_prefix, (best_b, best_nb) = max(beam.items(), key=lambda x: x[1][0] + x[1][1])
    total_conf = best_b + best_nb
    confidence = min(1.0, max(0.0, total_conf))
    return best_prefix, confidence
