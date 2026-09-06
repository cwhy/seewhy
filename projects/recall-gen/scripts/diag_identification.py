"""Why identification stops at 0.73 on the synthetic prior: geometry, not norm.

Report 20 explained the shortfall as precision — "to pick between two items
0.02 apart the output must land within 0.02, and it lands within 0.028" — and
concluded that precision tracks capacity. That arithmetic compares NORMS, and
identification does not depend on a norm.

Expanding the comparison the metric actually makes, with e = pred - target and
Delta = rival - target, both on hidden coordinates:

    ||pred - rival||^2 - ||pred - target||^2 = ||Delta||^2 - 2 e.Delta

so the rival wins iff  e.Delta / ||Delta||  >  ||Delta|| / 2.

Only the component of the error ALONG the target->rival direction matters. Every
component orthogonal to it cancels. `lib/splitfig.nearest_distractor` already
records that an isotropic-noise sweep left identification at 1.000 out to
sigma 1.1 for exactly this reason: over 416 coordinates a random error puts
~1/sqrt(416) of itself on any one direction.

So a network that fails identification is not merely imprecise. Its error is
POINTING somewhere. This script measures where.

    rho        e.Delta_hat / ||Delta||        > 0.5 means the rival wins
    cos_rival  e.Delta_hat / ||e||            1.0 = a pure two-item blend
    cos_cent   alignment with the context centroid instead
    iso        identification of target + isotropic noise of the SAME norm

and, inside the model, the effective retrieval weights the KDA state actually
applies. After the M context writes,

    S = sum_i  e_i  (x)  (k_i * A_i),      A_i = prod_{t>i} alpha_t

so the query's read is  o = sum_i e_i * w_i  with  w_i = (k_i * A_i).q / sqrt(dk).
Those w_i are this architecture's attention distribution, and whether argmax_i
w_i is the target separates ADDRESSING the right slot from RECONSTRUCTING it.

Analysis only — no training, no weights modified.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/diag_identification.py exp45 exp49 exp54
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT.parents[1]))          # repo root LAST — see workflow.md
sys.path.insert(0, str(PROJECT))

from lib.core import Cfg, ln, build_tokens, predict
from lib import splitfig
from rescore import rows as read_rows, run_from_row

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

COND = "A_seen_present"      # training worlds, answer present: report 20's 0.732


# ── instrumented forward ──────────────────────────────────────────────────────

def kda_trace(x, Lp, is_ctx, cfg):
    """`lib.core.kda`, additionally returning the per-item retrieval weights.

    Kept as a copy rather than a flag on the original because the original is
    on the training path and must stay exactly as every existing row was
    produced under. The forward arithmetic below is line-for-line the same.
    """
    B, N, D = x.shape
    H, DK = cfg.n_heads, cfg.dk
    sh = lambda t: t.reshape(B, N, H, DK).transpose(0, 2, 1, 3)

    q = sh(x @ Lp["Wq"])
    k = sh(x @ Lp["Wk"])
    v = sh(x @ Lp["Wv"])
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    alpha = jax.nn.sigmoid(sh(x @ Lp["Wa"] + Lp["ba"]))
    beta = jax.nn.sigmoid(x @ Lp["Wb"] + Lp["bb"]).transpose(0, 2, 1)

    gate = is_ctx[:, None, :]
    alpha = alpha * gate[..., None] + (1.0 - gate[..., None])
    beta = beta * gate

    def step(S, t):
        a_t, k_t, v_t, b_t = t
        S = S * a_t[:, :, None, :]
        vhat = jnp.einsum("bhvk,bhk->bhv", S, k_t)
        e = b_t[..., None] * (v_t - vhat)
        return S + jnp.einsum("bhv,bhk->bhvk", e, k_t), e

    seq = (alpha.transpose(2, 0, 1, 3), k.transpose(2, 0, 1, 3),
           v.transpose(2, 0, 1, 3), beta.transpose(2, 0, 1))
    S, e_seq = jax.lax.scan(step, jnp.zeros((B, H, DK, DK)), seq)
    e_all = e_seq.transpose(1, 2, 0, 3)                       # (B,H,N,DK)

    # A_i = prod over t>i of alpha_t, elementwise on the key axis. Query tokens
    # have alpha gated to 1, so they contribute nothing to this product.
    la = jnp.log(jnp.clip(alpha, 1e-9, None))                 # (B,H,N,DK)
    tail = jnp.cumsum(la[:, :, ::-1, :], axis=2)[:, :, ::-1, :] - la
    A = jnp.exp(tail)                                         # (B,H,N,DK)
    k_eff = k * A                                             # what survives to the end

    o = jnp.einsum("bhvk,bhnk->bhnv", S, q) / DK ** 0.5
    out = o.transpose(0, 2, 1, 3).reshape(B, N, D) @ Lp["Wo"]

    # w[b,h,n,i] : weight item i carries in token n's read. o_n = sum_i e_i w_i.
    w = jnp.einsum("bhik,bhnk->bhni", k_eff, q) / DK ** 0.5
    return out, w, e_all, k_eff


def forward_trace(p, pix, msk, is_ctx, cfg):
    x = pix @ p["W_pix"] + msk @ p["W_msk"]
    x = x + jnp.where(is_ctx[..., None] > 0.5, p["role"][0], p["role"][1])
    emb = x
    ws, es = [], []
    for Lp in p["layers"]:
        h, w, e_all, _ = kda_trace(ln(x, Lp["ln1_g"], Lp["ln1_b"]), Lp, is_ctx, cfg)
        ws.append(w)
        es.append(e_all)
        x = x + h
        x = x + (jax.nn.gelu(ln(x, Lp["ln2_g"], Lp["ln2_b"]) @ Lp["W1"] + Lp["b1"])
                 @ Lp["W2"] + Lp["b2"])
    out = jax.nn.sigmoid(ln(x, p["lnf_g"], p["lnf_b"]) @ p["head_W"] + p["head_b"])
    return out, ws, es, emb


# ── geometry of one episode ───────────────────────────────────────────────────

def episode_geometry(pred, ctx, qry, tgt_idx, mask):
    """Everything the identification comparison actually depends on.

    All distances are per-coordinate means over HIDDEN coordinates, the same
    units `masked_mse` and `nearest_distractor` use, so they are comparable to
    every nmse and margin already published.
    """
    m = mask > 0.5
    P = pred[:, m]                                   # (E, Dh)
    C = ctx[:, :, m]                                 # (E, M, Dh)
    T = qry[:, m]                                    # (E, Dh)
    E, M, Dh = C.shape
    ar = np.arange(E)

    d_pred = ((P[:, None, :] - C) ** 2).mean(-1)     # (E, M)
    pick = d_pred.argmin(-1)
    hit = pick == tgt_idx

    d_tgt = ((T[:, None, :] - C) ** 2).mean(-1)      # (E, M)
    d_tgt[ar, tgt_idx] = np.inf
    rival = d_tgt.argmin(-1)
    margin = d_tgt[ar, rival]                        # per-coordinate, unnormalised

    e = P - T                                        # the error vector
    err = (e ** 2).mean(-1)
    D = C[ar, rival] - T                             # target -> nearest rival
    nD = np.sqrt((D ** 2).sum(-1)) + 1e-12
    Dh_ = D / nD[:, None]
    ne = np.sqrt((e ** 2).sum(-1)) + 1e-12

    proj = (e * Dh_).sum(-1)                         # e . Delta_hat
    rho = proj / nD                                  # rival wins iff > 0.5
    cos_rival = proj / ne

    cent = C.mean(1)                                 # context centroid
    G = cent - T
    nG = np.sqrt((G ** 2).sum(-1)) + 1e-12
    cos_cent = (e * (G / nG[:, None])).sum(-1) / ne
    shrink = (e * (G / nG[:, None])).sum(-1) / nG    # fraction of the way to centroid

    # Does the WRONG pick tend to be the candidate nearest the centroid? That
    # would be shrinkage to the world mean rather than confusion with a rival.
    d_cent = ((cent[:, None, :] - C) ** 2).mean(-1)
    nearest_to_cent = d_cent.argmin(-1)

    return dict(P=P, hit=hit, pick=pick, rival=rival, margin=margin, err=err,
                rho=rho, cos_rival=cos_rival, cos_cent=cos_cent, shrink=shrink,
                pick_is_rival=(pick == rival),
                pick_is_centroid_nearest=(pick == nearest_to_cent),
                Dh=Dh_, T=T, C=C, ne=ne)


def sub(g, s):
    """The per-episode arrays a control needs, restricted to a subset of episodes."""
    return {k: g[k][s] for k in ("T", "C", "ne", "P")}


def isotropic_control(g, tgt_idx, rng, reps=4):
    """Identification of target + isotropic noise carrying the model's OWN norm.

    This is the control report 20's arithmetic implicitly assumes: if a norm of
    0.028 were enough to lose a 0.02 margin, this would fail too.
    """
    T, C, ne = g["T"], g["C"], g["ne"]
    E, M, Dh = C.shape
    acc = []
    for _ in range(reps):
        n = rng.standard_normal((E, Dh)).astype(np.float32)
        n = n / (np.sqrt((n ** 2).sum(-1, keepdims=True)) + 1e-12) * ne[:, None]
        d = (((T + n)[:, None, :] - C) ** 2).mean(-1)
        acc.append(float((d.argmin(-1) == tgt_idx).mean()))
    return float(np.mean(acc))


def aligned_control(g, tgt_idx, lam):
    """Identification of a pure two-item blend: target + lam * (rival - target).

    The opposite extreme from isotropic. Crosses at lam = 0.5 by construction;
    included so the measured rho can be read against a known curve.
    """
    T, C, Dh_ = g["T"], g["C"], g["Dh"]
    ar = np.arange(C.shape[0])
    nD = np.sqrt((((C[ar, g["rival"]] - T)) ** 2).sum(-1))
    P = T + lam * nD[:, None] * Dh_
    d = ((P[:, None, :] - C) ** 2).mean(-1)
    return float((d.argmin(-1) == tgt_idx).mean())


def deshrink_sweep(g, tgt_idx, gains=(1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)):
    """Push the output AWAY from the context centroid and re-identify.

    If the error is shrinkage toward the centroid, the target's identity is
    still present in the output's direction and only its magnitude is wrong.
    Rescaling about the centroid is a one-parameter, weight-free correction:

        pred' = centroid + gain * (pred - centroid)

    A gain that lifts identification well above the model's own score proves the
    information was never lost — the readout is simply calibrated for squared
    error, which pays for hedging, rather than for the argmin the metric takes.
    """
    C, P = g["C"], g["P"]
    cent = C.mean(1)
    out = {}
    for a in gains:
        Q = cent + a * (P - cent)
        d = ((Q[:, None, :] - C) ** 2).mean(-1)
        out[f"{a:.2f}"] = float((d.argmin(-1) == tgt_idx).mean())
    return out


# ── addressing ────────────────────────────────────────────────────────────────

def addressing(wqs, tgt_idx, rival):
    """Per layer: does the state's own retrieval weight point at the target?

    `wqs[L]` is (E, H, M) — the query token's weight on each context item.
    """
    out = []
    ar = np.arange(len(tgt_idx))
    for L, wq in enumerate(wqs):
        am = wq.argmax(-1)                                    # (E,H)
        hit = (am == tgt_idx[:, None]).mean(-1)               # per episode, over heads
        wt = wq[ar[:, None], np.arange(wq.shape[1])[None, :], tgt_idx[:, None]]
        wr = wq[ar[:, None], np.arange(wq.shape[1])[None, :], rival[:, None]]
        sd = wq.std(-1) + 1e-12
        out.append(dict(layer=L,
                        head_hit=hit,                          # (E,)
                        any_head_hit=(am == tgt_idx[:, None]).any(-1),
                        z_gap=((wt - wr) / sd).mean(-1)))      # (E,)
    return out


def oracle_by_episode(g, s, tgt_idx):
    """Identification of the EXACT answer, on this subset. The per-quartile
    ceiling: below 1.0 only where two context items share a hidden half."""
    T, C = g["T"][s], g["C"][s]
    d = ((T[:, None, :] - C) ** 2).mean(-1)
    return float((d.argmin(-1) == tgt_idx[s]).mean())


# ── driver ────────────────────────────────────────────────────────────────────

def quartiles(margin):
    q = np.quantile(margin, [0.25, 0.5, 0.75])
    return np.digitize(margin, q)


def analyse(exp, row, n_eval=512, chunk=128):
    rn = run_from_row(row)
    cfg = rn.cfg
    ctx_mode = row.get("ctx_mode", "iid")
    train_cls = tuple(rn.train_digits) if rn.train_digits else None
    held_cls = tuple(rn.held_digits) if rn.held_digits else None

    _, mask, ev = splitfig.build(rn.domain, rn.mask_rows, train_cls, held_cls, cfg,
                                 ctx_mode=ctx_mode, Q=rn.Q, M=rn.M, n_eval=n_eval)
    es = ev[COND]
    params = splitfig.load_params(exp)
    mask_j = jnp.array(mask)

    ctx = np.asarray(es.ctx)
    qry = np.asarray(es.qry)[:, 0]                    # query 0, as nearest_distractor uses
    tgt = np.asarray(es.tgt_idx)[:, 0]

    preds, all_w = [], None
    for i in range(0, ctx.shape[0], chunk):
        c = es.ctx[i:i + chunk]
        q = es.qry[i:i + chunk]
        pix, msk, is_ctx = build_tokens(c, q, mask_j)
        out, ws, _, _ = forward_trace(params, pix, msk, is_ctx, cfg)
        preds.append(np.asarray(out[:, rn.M, :]))     # first query token
        ws = [np.asarray(w[:, :, rn.M:rn.M + 1, :rn.M]) for w in ws]
        all_w = ws if all_w is None else [np.concatenate([a, b]) for a, b in zip(all_w, ws)]
    pred = np.concatenate(preds)
    all_w = [w[:, :, 0, :] for w in all_w]            # (E,H,M)

    g = episode_geometry(pred, ctx, qry, tgt, mask)
    mse_mean = es.mse_mean
    rng = np.random.default_rng(0)

    qi = quartiles(g["margin"])
    rows_out = []
    for qq in range(4):
        s = qi == qq
        rows_out.append(dict(
            quartile=qq, n=int(s.sum()),
            margin=float(g["margin"][s].mean() / mse_mean),
            err=float(g["err"][s].mean() / mse_mean),
            id_acc=float(g["hit"][s].mean()),
            rho=float(np.median(g["rho"][s])),
            frac_rho_gt_half=float((g["rho"][s] > 0.5).mean()),
            cos_rival=float(np.median(g["cos_rival"][s])),
            cos_cent=float(np.median(g["cos_cent"][s])),
            shrink=float(np.median(g["shrink"][s])),
            pick_is_rival=float(g["pick_is_rival"][s][~g["hit"][s]].mean())
                          if (~g["hit"][s]).sum() else float("nan"),
            pick_is_centroid_nearest=float(
                g["pick_is_centroid_nearest"][s][~g["hit"][s]].mean())
                if (~g["hit"][s]).sum() else float("nan"),
            iso=isotropic_control(sub(g, s), tgt[s], rng),
            oracle=float(oracle_by_episode(g, s, tgt)),
        ))

    addr = addressing(all_w, tgt, g["rival"])
    addr_rows = [dict(layer=a["layer"],
                      head_hit=float(a["head_hit"].mean()),
                      any_head_hit=float(a["any_head_hit"].mean()),
                      z_gap=float(a["z_gap"].mean()),
                      head_hit_q0=float(a["head_hit"][qi == 0].mean()),
                      any_head_hit_q0=float(a["any_head_hit"][qi == 0].mean()))
                 for a in addr]

    ctrl = dict(
        model_id=float(g["hit"].mean()),
        oracle_id=float(splitfig.oracle_id_acc(ev, mask, COND)),
        isotropic_same_norm=isotropic_control(g, tgt, rng),
        isotropic_q0=isotropic_control(sub(g, qi == 0), tgt[qi == 0], rng),
        blend_lam={f"{l:.2f}": aligned_control(g, tgt, l)
                   for l in (0.2, 0.4, 0.5, 0.6, 0.8)},
        deshrink=deshrink_sweep(g, tgt),
        deshrink_q0=deshrink_sweep(sub(g, qi == 0), tgt[qi == 0]),
    )
    return dict(experiment=exp, ctx_mode=ctx_mode, cfg=dict(cfg._asdict()),
                state_floats=cfg.state_floats, n_eval=n_eval,
                mse_mean=mse_mean, quartiles=rows_out, addressing=addr_rows,
                controls=ctrl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--out", default=str(PROJECT / "diag_identification.json"))
    a = ap.parse_args()

    by_exp = {r["experiment"]: r for r in read_rows()}
    out = []
    for exp in a.exps:
        logging.info(f"── {exp}")
        r = analyse(exp, by_exp[exp], n_eval=a.n_eval)
        out.append(r)
        c = r["controls"]
        logging.info(f"  model {c['model_id']:.3f}  oracle {c['oracle_id']:.3f}  "
                     f"isotropic(same norm) {c['isotropic_same_norm']:.3f}  "
                     f"isotropic on q0 {c['isotropic_q0']:.3f}")
        logging.info("  de-shrink gain -> id_acc (all / q0):")
        for gg in c["deshrink"]:
            logging.info(f"    x{gg}  {c['deshrink'][gg]:.3f}   {c['deshrink_q0'][gg]:.3f}")
        logging.info("  quartile   margin    err   id_acc   rho  P(rho>.5)  cos_riv  cos_cent")
        for q in r["quartiles"]:
            logging.info(f"    {q['quartile']}      {q['margin']:.3f}  {q['err']:.3f}  "
                         f"{q['id_acc']:.3f}  {q['rho']:+.3f}   {q['frac_rho_gt_half']:.3f}    "
                         f"{q['cos_rival']:+.3f}   {q['cos_cent']:+.3f}")
        logging.info("  layer  head_hit  any_head  z_gap   head_hit_q0  any_q0")
        for x in r["addressing"]:
            logging.info(f"    {x['layer']}     {x['head_hit']:.3f}     {x['any_head_hit']:.3f}"
                         f"   {x['z_gap']:+.2f}     {x['head_hit_q0']:.3f}      "
                         f"{x['any_head_hit_q0']:.3f}")
    Path(a.out).write_text(json.dumps(out, indent=2))
    logging.info(f"wrote {a.out}")


if __name__ == "__main__":
    main()
