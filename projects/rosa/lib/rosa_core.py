"""
ROSA core for integer-token streams.

A pure-Python port of bcml-labs/rosa-plus's `ROSACharPredictor` + `ROSAFallbackLM`
that operates on integer token IDs instead of characters. The SAM logic is
identical (Ukkonen-style online suffix automaton); the only change is that
transition dicts go from `dict[str, int]` to `dict[int, int]`. The Witten–Bell
fallback LM uses the same suffix-link interpolation.

We drop:
  - text generation (`_generate_mixed`, sampling utilities)
  - GRU neural adapter
  - char-level EOT handling
  - tqdm progress bars (caller can wrap if it wants progress)

We add:
  - `ROSACore.commit_burst(tokens)` — append a sample's tokens, mark boundary
  - `ROSACore.predict_argmax_along_stream(tokens)` — walk the SAM along a
    stream and at each position return the argmax predicted next-token plus
    a tag {"sam","lm","unigram","empty"} for the rosa_hit_rate metric

Usage:
    rosa = ROSACore(vocab_size=1024)
    rosa.commit_burst([0, 12, 47, 1, 8])           # <SAMPLE> tok tok <LABEL> y
    preds = rosa.predict_argmax_along_stream([0, 12, 47, 1])
    # preds[i] = (predicted_id, source_tag) for the position right after token i
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


# ── SAM: integer-token suffix automaton ───────────────────────────────────────


class ROSAIntPredictor:
    """Online/streaming SAM, boundary-aware. Identical logic to
    bcml-labs/rosa-plus's `ROSACharPredictor` but with integer transitions.

    Internal arrays (1 entry per state):
      b: list of dict[int, int]  — transitions (token -> next state)
      c: list[int]               — suffix links
      d: list[int]               — max length
      e: list[int]               — rightmost end position
      g: int                     — last/active state
    """

    def __init__(self) -> None:
        self.b: List[Dict[int, int]] = [{}]
        self.c: List[int] = [-1]
        self.d: List[int] = [0]
        self.e: List[int] = [-1]
        self.g: int = 0
        self.text: List[int] = []
        self.boundary_after: List[bool] = []

    def feed(self, tok: int) -> None:
        """Extend the SAM with one token (online construction)."""
        i = len(self.text)
        self.text.append(tok)
        if len(self.boundary_after) < len(self.text):
            self.boundary_after.append(False)

        b, c, d, e = self.b, self.c, self.d, self.e
        g = self.g

        r = len(b)
        b.append({})
        c.append(0)
        d.append(d[g] + 1)
        e.append(-1)
        p = g
        while p != -1 and tok not in b[p]:
            b[p][tok] = r
            p = c[p]
        if p == -1:
            c[r] = 0
        else:
            q = b[p][tok]
            if d[p] + 1 == d[q]:
                c[r] = q
            else:
                u = len(b)
                b.append(b[q].copy())
                c.append(c[q])
                d.append(d[p] + 1)
                e.append(e[q])
                while p != -1 and b[p].get(tok) == q:
                    b[p][tok] = u
                    p = c[p]
                c[q] = c[r] = u
        self.g = r

        v = self.g
        while v != -1 and self.e[v] < i:
            self.e[v] = i
            v = self.c[v]

    def mark_boundary(self) -> None:
        """Mark an example boundary after the most recently fed token."""
        if self.text:
            self.boundary_after[len(self.text) - 1] = True
        self.g = 0  # next example starts fresh from the root

    def to_state_dict(self) -> dict:
        return {
            "b": [list(d.items()) for d in self.b],
            "c": self.c,
            "d": self.d,
            "e": self.e,
            "g": self.g,
            "text": self.text,
            "boundary_after": self.boundary_after,
        }

    @classmethod
    def from_state_dict(cls, obj: dict) -> "ROSAIntPredictor":
        inst = cls()
        inst.b = [dict(items) for items in obj["b"]]
        inst.c = list(obj["c"])
        inst.d = list(obj["d"])
        inst.e = list(obj["e"])
        inst.g = int(obj.get("g", 0))
        inst.text = [int(t) for t in obj.get("text", [])]
        inst.boundary_after = list(obj.get("boundary_after", [False] * len(inst.text)))
        if len(inst.boundary_after) < len(inst.text):
            inst.boundary_after += [False] * (len(inst.text) - len(inst.boundary_after))
        return inst


def _advance_state(b: List[Dict[int, int]], c: List[int],
                   v: int, tok: int) -> int:
    """Walk a frozen SAM with token `tok` from state `v`. Returns the new state."""
    while v != -1 and tok not in b[v]:
        v = c[v]
    if v == -1:
        return b[0].get(tok, 0)
    return b[v][tok]


def _predict_deterministic(
    b: List[Dict[int, int]], c: List[int], d: List[int], e: List[int],
    train_text: List[int], v: int, boundary_after: List[bool],
) -> Optional[int]:
    """Deterministic ROSA prediction: read the token right after the rightmost
    occurrence of the current suffix. Refuses to cross example boundaries.
    Returns None if no answer."""
    u = v
    n = len(train_text)
    while u != -1:
        i = e[u]
        j = i + 1
        if d[u] > 0 and 0 <= j < n:
            if 0 <= i < len(boundary_after) and boundary_after[i]:
                u = c[u]
                continue
            return train_text[j]
        u = c[u]
    return None


# ── Witten–Bell fallback over the suffix-link chain ───────────────────────────


class ROSAFallbackLM:
    """Witten–Bell interpolation LM down the SAM suffix chain. Counts are
    accumulated per-example; pairs do not cross boundaries."""

    def __init__(
        self,
        sam: ROSAIntPredictor,
        examples: List[List[int]],
        *,
        max_order: Optional[int] = None,
    ) -> None:
        self.b, self.c, self.d, self.e = sam.b, sam.c, sam.d, sam.e
        self.max_order = max_order
        flat: List[int] = [t for ex in examples for t in ex]
        self.alphabet: List[int] = sorted(set(flat)) if flat else [0]
        self.unigram: Counter = Counter(flat) if flat else Counter({0: 1})
        self.freq: List[Dict[int, int]] = [defaultdict(int) for _ in range(len(self.b))]
        self._cache: Dict[int, Dict[int, float]] = {}
        self._build_counts(examples)

    def _build_counts(self, examples: List[List[int]]) -> None:
        b, c, d = self.b, self.c, self.d
        freq = self.freq
        max_order = self.max_order

        for seg in examples:
            if not seg:
                continue
            v = 0
            for i in range(len(seg) - 1):
                tok = seg[i]
                u = v
                while u != -1 and tok not in b[u]:
                    u = c[u]
                v = b[0].get(tok, 0) if u == -1 else b[u][tok]

                ctx = v
                if max_order is not None:
                    while ctx != -1 and d[ctx] > max_order:
                        ctx = c[ctx]
                    if ctx == -1:
                        ctx = 0
                freq[ctx][seg[i + 1]] += 1

        self._propagate_counts_up_suffix_links()
        self._cache.clear()

    def _propagate_counts_up_suffix_links(self) -> None:
        """After filling counts at the longest contexts, push them up the
        suffix-link tree so every shorter context has aggregated counts."""
        order = sorted(range(len(self.b)), key=lambda v: self.d[v], reverse=True)
        for v in order:
            p = self.c[v]
            if p < 0:
                continue
            if not self.freq[v]:
                continue
            dv = self.freq[v]
            dp = self.freq[p]
            for tok, cnt in dv.items():
                dp[tok] += cnt

    def ensure_capacity(self) -> None:
        missing = len(self.b) - len(self.freq)
        if missing > 0:
            self.freq.extend(defaultdict(int) for _ in range(missing))

    def observe_pair(self, ctx_state: int, next_tok: int, *, propagate: bool = True) -> None:
        """Online update: record one (ctx_state -> next_tok) observation."""
        if next_tok not in self.alphabet:
            return
        self.ensure_capacity()
        self.freq[ctx_state][next_tok] += 1
        if propagate:
            u = self.c[ctx_state]
            while u != -1:
                self.freq[u][next_tok] += 1
                u = self.c[u]
        u = ctx_state
        while u != -1:
            self._cache.pop(u, None)
            u = self.c[u]

    def probs_for_state(self, v: int) -> Dict[int, float]:
        if v in self._cache:
            return self._cache[v]

        chain: List[int] = []
        u = v
        while u != -1:
            if self.max_order is not None and self.d[u] > self.max_order:
                u = self.c[u]
                continue
            chain.append(u)
            u = self.c[u]

        residual = 1.0
        probs: Dict[int, float] = {}

        def add_counts(state: int, scale: float) -> None:
            if scale <= 0.0:
                return
            total = sum(self.freq[state].values())
            if total == 0:
                return
            inv_total = 1.0 / total
            for tok, cnt in self.freq[state].items():
                probs[tok] = probs.get(tok, 0.0) + scale * (cnt * inv_total)

        for state in chain:
            N = sum(self.freq[state].values())
            T = len(self.freq[state])
            if N == 0:
                continue
            lam = N / (N + T) if T > 0 else 1.0
            add_counts(state, residual * lam)
            residual *= 1.0 - lam

        total_uni = sum(self.unigram.values())
        if total_uni > 0 and residual > 0.0:
            inv_total = 1.0 / total_uni
            for tok, cnt in self.unigram.items():
                probs[tok] = probs.get(tok, 0.0) + residual * (cnt * inv_total)

        s = sum(probs.values())
        if s > 0:
            inv_s = 1.0 / s
            for k in list(probs.keys()):
                probs[k] *= inv_s
        else:
            inv_a = 1.0 / max(1, len(self.alphabet))
            probs = {tok: inv_a for tok in self.alphabet}

        self._cache[v] = probs
        return probs

    def to_state_dict(self) -> dict:
        return {
            "alphabet": self.alphabet,
            "unigram": dict(self.unigram),
            "freq": [dict(d) for d in self.freq],
            "max_order": self.max_order,
        }

    @classmethod
    def from_state_dict(cls, sam: ROSAIntPredictor, obj: dict) -> "ROSAFallbackLM":
        inst = cls.__new__(cls)
        inst.b, inst.c, inst.d, inst.e = sam.b, sam.c, sam.d, sam.e
        inst.max_order = obj.get("max_order", None)
        inst.alphabet = [int(t) for t in obj["alphabet"]]
        inst.unigram = Counter({int(k): int(v) for k, v in obj["unigram"].items()})
        inst.freq = [defaultdict(int, {int(k): int(v) for k, v in d.items()}) for d in obj["freq"]]
        inst._cache = {}
        return inst


# ── High-level wrapper ────────────────────────────────────────────────────────

PredSource = str  # one of {"sam", "lm", "unigram", "empty"}


class ROSACore:
    """High-level continual ROSA predictor for integer token streams.

    Flow:
        rosa = ROSACore(vocab_size=1024)
        for batch in epoch:
            for stream in batch:
                preds = rosa.predict_argmax_along_stream(stream)   # for forward pass
            ... do JAX forward / backprop ...
            for stream in batch:
                rosa.commit_burst(stream)                          # update SAM
        rosa.build_lm(all_examples)   # optional; or call observe_pair online
    """

    def __init__(self, vocab_size: int, *, max_order: Optional[int] = None) -> None:
        self.vocab_size = vocab_size
        self.max_order = max_order
        self.sam = ROSAIntPredictor()
        self.lm: Optional[ROSAFallbackLM] = None
        self._n_committed_bursts: int = 0
        # Running unigram counts, kept incrementally so predict_argmax_at_state
        # doesn't have to rebuild Counter(sam.text) on every fallback call.
        self._unigram: Counter = Counter()
        self._unigram_top: Optional[int] = None  # cached argmax over unigram

    # ---- update ----

    def commit_burst(self, tokens: List[int]) -> None:
        """Append one sample's burst of tokens to the SAM and mark a boundary."""
        for t in tokens:
            t_int = int(t)
            self.sam.feed(t_int)
            self._unigram[t_int] += 1
        self.sam.mark_boundary()
        self._n_committed_bursts += 1
        self._unigram_top = None  # invalidate cache

    def commit_batch(self, batch_tokens: List[List[int]]) -> None:
        for tokens in batch_tokens:
            self.commit_burst(tokens)

    def build_lm(self, examples: List[List[int]]) -> None:
        """(Re)build the Witten–Bell fallback LM from a list of full examples."""
        self.lm = ROSAFallbackLM(self.sam, examples, max_order=self.max_order)

    # ---- query ----

    def predict_argmax_at_state(self, v: int) -> Tuple[int, PredSource]:
        """Argmax-style next-token prediction at SAM state `v`.
        Returns (token_id, source_tag) where source_tag is one of
        'sam' (deterministic), 'lm' (Witten-Bell argmax), 'unigram', 'empty'."""
        sam = self.sam
        det = _predict_deterministic(sam.b, sam.c, sam.d, sam.e,
                                     sam.text, v, sam.boundary_after)
        if det is not None:
            return int(det), "sam"

        if self.lm is not None:
            probs = self.lm.probs_for_state(v)
            if probs:
                tok = max(probs.items(), key=lambda kv: kv[1])[0]
                return int(tok), "lm"

        if self._unigram:
            if self._unigram_top is None:
                self._unigram_top = self._unigram.most_common(1)[0][0]
            return int(self._unigram_top), "unigram"

        return 0, "empty"

    def predict_argmax_along_stream(
        self, tokens: List[int],
    ) -> List[Tuple[int, PredSource]]:
        """Walk the SAM along `tokens` and return a prediction at each position.

        `tokens` is the stream of *observed* tokens (e.g.
        `<SAMPLE> s_1 ... s_K <LABEL>`); the i-th entry of the returned list is
        the predicted next-token *after* observing `tokens[:i+1]`.
        """
        b, c = self.sam.b, self.sam.c
        v = 0
        out: List[Tuple[int, PredSource]] = []
        for tok in tokens:
            v = _advance_state(b, c, v, int(tok))
            out.append(self.predict_argmax_at_state(v))
        return out

    # ---- diagnostics ----

    def n_states(self) -> int:
        return len(self.sam.b)

    def n_tokens(self) -> int:
        return len(self.sam.text)

    def n_bursts(self) -> int:
        return self._n_committed_bursts

    # ---- persistence ----

    def save(self, path: str) -> None:
        payload = {
            "vocab_size": self.vocab_size,
            "max_order": self.max_order,
            "n_committed_bursts": self._n_committed_bursts,
            "sam": self.sam.to_state_dict(),
            "lm": self.lm.to_state_dict() if self.lm is not None else None,
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "ROSACore":
        with open(path) as f:
            payload = json.load(f)
        inst = cls(vocab_size=int(payload["vocab_size"]),
                   max_order=payload.get("max_order"))
        inst._n_committed_bursts = int(payload.get("n_committed_bursts", 0))
        inst.sam = ROSAIntPredictor.from_state_dict(payload["sam"])
        # Rebuild running unigram counts (one-time pass over sam.text)
        inst._unigram = Counter(inst.sam.text)
        inst._unigram_top = None
        if payload.get("lm") is not None:
            inst.lm = ROSAFallbackLM.from_state_dict(inst.sam, payload["lm"])
        return inst
