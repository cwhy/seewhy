from datasets import Dataset, DatasetDict, load_dataset
from typing import Literal, NamedTuple
import jax.numpy as jnp
from jax import Array
import numpy as np
import pickle
import os
from pathlib import Path


class SudokuDataset(NamedTuple):
    n_train: int
    n_test: int
    X_train: Array   # (n_train, 81) int32  — 0=empty, 1-9=given digit
    y_train: Array   # (n_train, 81) int32  — solution digits 1-9
    X_test: Array
    y_test: Array


class Supervised1D(NamedTuple):
    n_samples: int
    d_x: int
    d_y: int
    X: Array
    y: Array
    X_test: Array
    y_test: Array


class ImageClassification(NamedTuple):
    n_samples: int
    n_test_samples: int
    d_x: tuple[int, int]
    d_y: int
    n_channels: int
    X: Array
    y: Array
    X_test: Array
    y_test: Array


class FewShotImages(NamedTuple):
    """Two *class-disjoint* image sets, for few-shot / in-context learning.

    Unlike ImageClassification, the split is over CLASSES, not samples: no
    character in `y_ev` appears anywhere in `y_bg`. A model cannot answer an
    evaluation episode from a memorised class prototype, because it has never
    seen the class. `bg` (background) is for training, `ev` (evaluation) for
    testing.

    Character ids are contiguous within each split (`0..n_char_bg-1` and
    `0..n_char_ev-1`) — they index into that split only and are not comparable
    across splits. Alphabet ids stay on the shared 0..49 scale so the two
    splits' alphabet inventories can be checked against each other.
    """

    d_x: tuple[int, int]
    n_channels: int
    X_bg: Array   # (n_bg, 1, size, size) uint8
    y_bg: Array   # (n_bg,) int32 — character id within the background split
    a_bg: Array   # (n_bg,) int32 — alphabet id (global)
    X_ev: Array
    y_ev: Array
    a_ev: Array
    n_char_bg: int
    n_char_ev: int


cache_dir = str(Path.home() / ".cache/huggingface/datasets")


def _get_cache_path(
    data: str, n_tr: int | None, n_tst: int | None, kind: str | None = None
) -> str:
    """Generate cache file path for the dataset configuration."""
    cache_base = Path(cache_dir).parent / "processed_datasets"
    cache_base.mkdir(exist_ok=True)

    # Create a unique filename based on dataset and parameters
    n_tr_str = f"n_tr_{n_tr}" if n_tr else "n_tr_all"
    n_tst_str = f"n_tst_{n_tst}" if n_tst else "n_tst_all"
    kind_suffix = f"_{kind}" if kind else ""
    filename = f"{data}{kind_suffix}_{n_tr_str}_{n_tst_str}.pkl"
    return str(cache_base / filename)


def _load_from_cache(cache_path: str):
    """Load a cached dataset object if it exists."""
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
    except (pickle.PickleError, EOFError, FileNotFoundError):
        # If cache is corrupted or doesn't exist, return None
        pass
    return None


def _save_to_cache(cache_path: str, dataset) -> None:
    """Save a dataset object to cache."""
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(dataset, f)
    except Exception as e:
        # Don't raise if caching fails - keep main functionality working
        print(f"Warning: Failed to cache dataset to {cache_path}: {e}")


def load_supervised_1d(
    data: Literal["mnist", "fashion_mnist"],
    n_tr: int | None = None,
    n_tst: int | None = None,
) -> Supervised1D:

    # Try to load from cache first (use kind to avoid collision with image datasets)
    cache_path = _get_cache_path(data, n_tr, n_tst, kind="1d")
    cached_dataset = _load_from_cache(cache_path)
    if cached_dataset is not None:
        return cached_dataset

    if data == "mnist":
        ds = load_dataset("mnist", cache_dir=cache_dir).with_format("jax")
    elif data == "fashion_mnist":
        ds = load_dataset("fashion_mnist", cache_dir=cache_dir).with_format("jax")
    else:  # pragma: no cover - guarded by Literal but kept for safety
        raise ValueError(
            "Unsupported 1D dataset "
            f"'{data}'. Supported values are: 'mnist', 'fashion_mnist'."
        )

    assert isinstance(ds, DatasetDict)
    train_ds = ds["train"]
    test_ds = ds["test"]

    # Access the data as JAX arrays via datasets' JAX formatting
    X_img: Array = jnp.array(train_ds["image"][:])
    y: Array = jnp.array(train_ds["label"][:])
    X_img_test: Array = jnp.array(test_ds["image"][:])
    y_test: Array = jnp.array(test_ds["label"][:])

    # Apply slicing after accessing the data
    if n_tr:
        X_img = X_img[:n_tr]
        y = y[:n_tr]
    if n_tst:
        X_img_test = X_img_test[:n_tst]
        y_test = y_test[:n_tst]

    n_samples = X_img.shape[0]
    X = X_img.reshape((n_samples, -1))
    n_test_samples = X_img_test.shape[0]
    X_test = X_img_test.reshape((n_test_samples, -1))
    n_samples, d_x = X.shape
    d_y = len(set(y.tolist()))
    dataset = Supervised1D(n_samples, d_x, d_y, X, y, X_test, y_test)
    _save_to_cache(cache_path, dataset)
    return dataset


def load_supervised_image(
    data: Literal["mnist", "fashion_mnist", "cifar10"],
    n_tr: int | None = None,
    n_tst: int | None = None,
) -> ImageClassification:

    # Try to load from cache first
    cache_path = _get_cache_path(data, n_tr, n_tst)
    cached_dataset = _load_from_cache(cache_path)
    if cached_dataset is not None:
        return cached_dataset

    # If not in cache, load and process the dataset
    if data == "mnist":
        ds = load_dataset("mnist", cache_dir=cache_dir).with_format("jax")

        assert isinstance(ds, DatasetDict)
        train_ds = ds["train"]
        test_ds = ds["test"]

        # Access the data as JAX arrays
        X_img = jnp.array(train_ds["image"][:])
        y = jnp.array(train_ds["label"][:])
        X_img_test = jnp.array(test_ds["image"][:])
        y_test = jnp.array(test_ds["label"][:])

        # Apply slicing after accessing the data
        if n_tr:
            X_img = X_img[:n_tr]
            y = y[:n_tr]
        if n_tst:
            X_img_test = X_img_test[:n_tst]
            y_test = y_test[:n_tst]

        n_samples = X_img.shape[0]
        n_test_samples = X_img_test.shape[0]
        X_train = X_img.reshape((n_samples, 1, 28, 28))
        X_test = X_img_test.reshape((n_test_samples, 1, 28, 28))
        n_channels = 1
        d_x = (28, 28)
        d_y = len(set(y.tolist()))
        dataset = ImageClassification(
            n_samples,
            n_test_samples,
            d_x,
            d_y,
            n_channels,
            X_train,
            y,
            X_test,
            y_test,
        )
        _save_to_cache(cache_path, dataset)
        return dataset
    elif data == "fashion_mnist":
        ds = load_dataset("fashion_mnist", cache_dir=cache_dir).with_format("jax")

        assert isinstance(ds, DatasetDict)
        train_ds = ds["train"]
        test_ds = ds["test"]

        # Access the data as JAX arrays
        X_img= jnp.array(train_ds["image"][:])
        y = jnp.array(train_ds["label"][:])
        X_img_test = jnp.array(test_ds["image"][:])
        y_test = jnp.array(test_ds["label"][:])

        # Apply slicing after accessing the data
        if n_tr:
            X_img = X_img[:n_tr]
            y = y[:n_tr]
        if n_tst:
            X_img_test = X_img_test[:n_tst]
            y_test = y_test[:n_tst]

        n_samples = X_img.shape[0]
        n_test_samples = X_img_test.shape[0]
        X_train = X_img.reshape((n_samples, 1, 28, 28))
        X_test = X_img_test.reshape((n_test_samples, 1, 28, 28))
        n_channels = 1
        d_x = (28, 28)
        d_y = len(set(y.tolist()))
        dataset = ImageClassification(
            n_samples,
            n_test_samples,
            d_x,
            d_y,
            n_channels,
            X_train,
            y,
            X_test,
            y_test,
        )
        _save_to_cache(cache_path, dataset)
        return dataset
    elif data == "cifar10":
        import numpy as np
        ds = load_dataset("uoft-cs/cifar10", cache_dir=cache_dir)

        assert isinstance(ds, DatasetDict)
        train_ds = ds["train"]
        test_ds = ds["test"]

        # uoft-cs/cifar10 uses column "img" (PIL Images) — convert via numpy
        X_img: Array = jnp.array(np.stack([np.array(img) for img in train_ds["img"]]))
        y: Array = jnp.array(train_ds["label"])
        X_img_test: Array = jnp.array(np.stack([np.array(img) for img in test_ds["img"]]))
        y_test: Array = jnp.array(test_ds["label"])

        # Apply slicing after accessing the data
        if n_tr:
            X_img = X_img[:n_tr]
            y = y[:n_tr]
        if n_tst:
            X_img_test = X_img_test[:n_tst]
            y_test = y_test[:n_tst]

        n_samples = X_img.shape[0]
        n_test_samples = X_img_test.shape[0]
        X_train = X_img.transpose((0, 3, 1, 2))
        X_test = X_img_test.transpose((0, 3, 1, 2))
        n_channels = 3
        d_x = (32, 32)
        d_y = len(set(y.tolist()))
        dataset = ImageClassification(
            n_samples,
            n_test_samples,
            d_x,
            d_y,
            n_channels,
            X_train,
            y,
            X_test,
            y_test,
        )
        _save_to_cache(cache_path, dataset)
        return dataset
    else:
        raise ValueError(
            "Unsupported image dataset "
            f"'{data}'. Supported values are: 'mnist', 'fashion_mnist', 'cifar10'."
        )


def load_omniglot(
    size: int = 28,
    invert: bool = True,
    n_bg: int | None = None,
    n_ev: int | None = None,
) -> FewShotImages:
    """Load Omniglot from `dpdl-benchmark/omniglot` as class-disjoint splits.

    The two HuggingFace splits are Lake et al.'s originals: `train` is the
    *background* set (964 characters from 30 alphabets, 20 drawings each) and
    `test` is the *evaluation* set (659 characters from 20 alphabets). The
    character inventories are disjoint, which is the whole point — see
    FewShotImages.

    Args:
        size:   edge length to resize to (105 native; 28 matches MNIST).
        invert: Omniglot ships black strokes on white. Inverting puts the ink
                at high values, matching MNIST, so a shared pixel-bin
                vocabulary means the same thing on both and "ink" is `> 0`.
        n_bg / n_ev: optional caps on images per split, for quick runs.
    """
    from PIL import Image

    kind = f"fewshot{size}{'_inv' if invert else ''}"
    cache_path = _get_cache_path("omniglot", n_bg, n_ev, kind=kind)
    cached = _load_from_cache(cache_path)
    if cached is not None:
        return cached

    ds = load_dataset("dpdl-benchmark/omniglot", cache_dir=cache_dir)
    assert isinstance(ds, DatasetDict)

    def to_arrays(split, n):
        rows = split if n is None else split.select(range(min(n, len(split))))
        imgs = np.stack([
            np.asarray(
                im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.uint8
            )
            for im in rows["image"]
        ])
        if invert:
            imgs = 255 - imgs
        return (
            imgs,
            np.asarray(rows["label"], dtype=np.int32),
            np.asarray(rows["alphabet"], dtype=np.int32),
        )

    X_bg, chars_bg, a_bg = to_arrays(ds["train"], n_bg)
    X_ev, chars_ev, a_ev = to_arrays(ds["test"], n_ev)

    overlap = set(chars_bg.tolist()) & set(chars_ev.tolist())
    if overlap:
        raise ValueError(
            f"background and evaluation splits share {len(overlap)} character "
            "ids — the class-disjointness this loader promises does not hold"
        )

    def contiguous(labels):
        uniq = np.unique(labels)
        remap = {int(v): i for i, v in enumerate(uniq)}
        return np.array([remap[int(v)] for v in labels], dtype=np.int32), len(uniq)

    y_bg, n_char_bg = contiguous(chars_bg)
    y_ev, n_char_ev = contiguous(chars_ev)

    result = FewShotImages(
        d_x=(size, size),
        n_channels=1,
        X_bg=jnp.asarray(X_bg.reshape(-1, 1, size, size)),
        y_bg=jnp.asarray(y_bg),
        a_bg=jnp.asarray(a_bg),
        X_ev=jnp.asarray(X_ev.reshape(-1, 1, size, size)),
        y_ev=jnp.asarray(y_ev),
        a_ev=jnp.asarray(a_ev),
        n_char_bg=n_char_bg,
        n_char_ev=n_char_ev,
    )
    _save_to_cache(cache_path, result)
    return result


def load_sudoku_extreme(
    n_tr: int | None = None,
    n_tst: int | None = None,
) -> SudokuDataset:
    """Load sapientinc/sudoku-extreme from HuggingFace.

    Puzzles are 81-char strings ('.' = empty, '1'-'9' = given digit).
    X: int32 arrays with 0=empty, 1-9=digit.
    y: int32 solution arrays with digits 1-9.
    """
    cache_path = _get_cache_path("sudoku_extreme", n_tr, n_tst)
    cached = _load_from_cache(cache_path)
    if cached is not None:
        return cached

    ds = load_dataset("sapientinc/sudoku-extreme", cache_dir=cache_dir)
    assert isinstance(ds, DatasetDict)

    def to_arrays(split, n):
        rows = split if n is None else split[:n]
        X = np.array(
            [[0 if c == "." else int(c) for c in q] for q in rows["question"]],
            dtype=np.int32,
        )
        y = np.array(
            [[int(c) for c in a] for a in rows["answer"]],
            dtype=np.int32,
        )
        return jnp.array(X), jnp.array(y)

    X_train, y_train = to_arrays(ds["train"], n_tr)
    X_test,  y_test  = to_arrays(ds["test"],  n_tst)

    result = SudokuDataset(
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    _save_to_cache(cache_path, result)
    return result


# ── Chess: Lichess Stockfish evaluations ──────────────────────────────────────
#
# Source file is a 7.5 GB SQLite database on the GPU box, one row per position:
#
#     CREATE TABLE evaluations(id, fen, binary, eval)
#
# 37 164 639 rows, ids dense from 1. `eval` is a Stockfish score in pawns from
# White's point of view; `binary` is an opaque bitboard blob this loader ignores
# in favour of parsing the FEN, which is self-describing.
#
# The loader below is deliberately RAW: it hands back FEN strings and scores,
# not tensors. Rows are consecutive plies of one game, so ids are strongly
# autocorrelated — positions 4642 and 4643 differ by a single move. Two
# consequences shape the interface:
#
#   * the train/test split is a single id THRESHOLD, so no game straddles it;
#   * within each side, positions are sampled uniformly at random rather than
#     taken as a contiguous block, so a pool is near-iid instead of being a few
#     hundred games in move order.

CHESS_DB_PATH = "/home/newuser/Projects/chess/2021-07-31-lichess-evaluations-37MM.db"
CHESS_N_ROWS = 37_164_639
CHESS_SPLIT_ID = 29_731_711        # ~80% of the table; games do not straddle it

# Channel order for `fen_to_planes`. Index 0 is the empty square, so a plane
# vector is a proper 13-way one-hot and its mean over a pool is the per-square
# marginal distribution over piece types.
CHESS_PIECES = " PNBRQKpnbrqk"


class ChessPositions(NamedTuple):
    """Raw positions. No encoding is applied — `fen` is the source string.

    `idx` is the row id in the source table, kept so a position can be traced
    back and so adjacency between two sampled positions is checkable.
    """

    n_train: int
    n_test: int
    fen: np.ndarray         # (n_train,) dtype object — FEN exactly as stored
    score: np.ndarray       # (n_train,) float32 — Stockfish eval in pawns, White POV
    idx: np.ndarray         # (n_train,) int64 — source row id
    fen_test: np.ndarray
    score_test: np.ndarray
    idx_test: np.ndarray


def _chess_sample(conn, lo: int, hi: int, n: int, rng, chunk: int = 5000):
    """Draw `n` distinct row ids uniformly from [lo, hi) and fetch them.

    Ids are dense, so a uniform draw over the range needs no index scan; the
    `IN (...)` lookups ride the table's `id_idx` and run at ~7k rows/s/chunk.
    """
    # Population passed as a count, not an array: materialising 30M int64 ids
    # only to permute them costs 240 MB and seconds, and Generator.choice takes
    # the cheap path when the population is an integer.
    ids = np.sort(rng.choice(hi - lo, size=n, replace=False).astype(np.int64) + lo)
    fens, scores, got = [], [], []
    for i in range(0, n, chunk):
        block = ids[i:i + chunk]
        q = ",".join(str(int(v)) for v in block)
        for rid, fen, ev in conn.execute(
                f"SELECT id, fen, eval FROM evaluations WHERE id IN ({q})"):
            got.append(rid)
            fens.append(fen)
            scores.append(float(ev) if ev is not None else 0.0)
    return (np.array(fens, dtype=object), np.array(scores, np.float32),
            np.array(got, np.int64))


def load_chess_positions(
    n_tr: int = 200_000,
    n_tst: int = 150_000,
    db_path: str | None = None,
    split_id: int = CHESS_SPLIT_ID,
    seed: int = 0,
) -> ChessPositions:
    """Raw FEN + Stockfish score, sampled from the Lichess evaluations database.

    Train positions come from ids below `split_id`, test positions from above
    it, so the two sides share no game.
    """
    import sqlite3

    cache_path = _get_cache_path("chess_evals", n_tr, n_tst, kind=f"raw_s{seed}")
    cached = _load_from_cache(cache_path)
    if cached is not None:
        return cached

    path = db_path or CHESS_DB_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Lichess evaluations database not found at {path}. It lives on the "
            "GPU box only; pass db_path= to point elsewhere."
        )
    conn = sqlite3.connect(path)
    rng = np.random.default_rng(seed)
    fen, score, idx = _chess_sample(conn, 1, split_id, n_tr, rng)
    fen_t, score_t, idx_t = _chess_sample(conn, split_id, CHESS_N_ROWS + 1, n_tst, rng)
    conn.close()

    ds = ChessPositions(
        n_train=len(fen), n_test=len(fen_t),
        fen=fen, score=score, idx=idx,
        fen_test=fen_t, score_test=score_t, idx_test=idx_t,
    )
    _save_to_cache(cache_path, ds)
    return ds


def fen_to_planes(fens) -> np.ndarray:
    """FEN strings -> (N, 8, 8, 13) float32 one-hot piece planes.

    Axis 0 is the rank in DISPLAY order — index 0 is rank 8, the top of a drawn
    board — and axis 1 is the file a..h. That ordering means a figure can
    `imshow` the array without transposing, and a mask over files is a slice on
    axis 1.

    Only piece placement is encoded. Side to move, castling rights and the
    en-passant square are dropped: this is the piece representation, and a
    completion task over hidden squares has no use for them.
    """
    fens = np.asarray(fens, dtype=object).reshape(-1)
    out = np.zeros((len(fens), 8, 8, 13), np.float32)
    out[..., 0] = 1.0                                   # every square empty
    lookup = {c: i for i, c in enumerate(CHESS_PIECES)}
    for n, fen in enumerate(fens):
        for r, row in enumerate(fen.split(" ", 1)[0].split("/")):
            f = 0
            for ch in row:
                if ch.isdigit():
                    f += int(ch)
                else:
                    out[n, r, f, 0] = 0.0
                    out[n, r, f, lookup[ch]] = 1.0
                    f += 1
    return out


def chess_piece_count(fens) -> np.ndarray:
    """Pieces on the board, per FEN — the phase axis this project splits on."""
    return np.array([sum(c.isalpha() for c in f.split(" ", 1)[0])
                     for f in np.asarray(fens, dtype=object).reshape(-1)], np.int32)
