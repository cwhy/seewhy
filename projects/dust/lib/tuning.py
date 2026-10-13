"""Pick tuned hyperparameters out of results.jsonl, the way the paper does.

Every cell is tuned on one seed (42) and judged by *validation* loss -- never test --
then the winner is repeated on more seeds. These helpers read the grid rows already in
`results.jsonl` and return the winning settings, so a ladder experiment can be launched
without anyone copying numbers by hand, and the choice is reproducible from the log.
"""

from lib import results as R


def best_row(method: str, budget: str, pop: int, exps: tuple[str, ...], seed: int = 42) -> dict | None:
    rows = [r for r in R.train_rows(R.load(), method=method, budget=budget, pop=pop, seed=seed)
            if r["experiment"] in exps and r["best_val"] == r["best_val"]]
    return min(rows, key=lambda r: r["best_val"]) if rows else None


def eggroll_settings(budget: str, pop: int, exps=("exp5", "exp8", "exp14"), fallback_pop: int | None = None) -> dict:
    """lr, sigma, beta1, beta2 of the best-validation EGGROLL grid cell at this population.

    If the population was never tuned (too expensive), fall back to `fallback_pop`'s winner
    and say so in the row's tag.
    """
    r = best_row("eggroll", budget, pop, exps)
    tag = ""
    if r is None and fallback_pop is not None:
        r = best_row("eggroll", budget, fallback_pop, exps)
        tag = f"lr_from_{fallback_pop}"
    if r is None:
        raise LookupError(f"no tuned EGGROLL cell for {budget} / {pop}")
    return dict(lr=r["lr"], sigma=r["sigma"], momentum=r["momentum"], beta2=r["beta2"], tag=tag)


def best_opt_row(method: str, pop: int, opt: str, exps: tuple[str, ...], budget: str = "1M", seed: int = 42) -> dict | None:
    """Best-validation grid cell under one optimiser (rows carry a top-level `opt`; older SGD rows do not)."""
    rows = [r for r in R.train_rows(R.load(), method=method, budget=budget, pop=pop, seed=seed)
            if r["experiment"] in exps and r.get("opt", "sgd") == opt and r["best_val"] == r["best_val"]]
    return min(rows, key=lambda r: r["best_val"]) if rows else None


def optimiser_settings(method: str, pop: int, opt: str, exps: tuple[str, ...], fallback_pop: int | None = None) -> dict:
    """lr (and sigma for EGGROLL) of the best-validation cell under Adam or Muon; falls back to another
    population's winner where that population was too expensive to tune, and tags the row so."""
    r, tag = best_opt_row(method, pop, opt, exps), f"{opt}_tuned"
    if r is None and fallback_pop is not None:
        r, tag = best_opt_row(method, fallback_pop, opt, exps), f"{opt}_lr_from_{fallback_pop}"
    if r is None:
        raise LookupError(f"no tuned {opt} cell for {method} / {pop}")
    out = dict(lr=r["lr"], opt=opt, tag=tag)
    if method == "eggroll":
        out["sigma"] = r["sigma"]
    return out


def dust_settings(budget: str, pop: int, exps=("exp4", "exp11")) -> dict:
    """lr, momentum of the best-validation DUST grid cell (populations without released settings)."""
    r = best_row("dust", budget, pop, exps)
    if r is None:
        raise LookupError(f"no tuned DUST cell for {budget} / {pop}")
    return dict(lr=r["lr"], momentum=r["momentum"])
