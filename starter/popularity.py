"""Per-product popularity features, shared by training and serving.

    POP = load_popularity(query_fn, until=CUT)   # training: only history BEFORE the cut (no leakage)
    POP = load_popularity(query_fn)              # serving: all history  (saved once to starter/popularity.pkl)

`query_fn(sql)` must return a DataFrame (the notebook's query() and recommender's query_df() both do).
Result: {product_id: {feature: value}} plus a "__default__" entry for products we have no stats for.
"""
from __future__ import annotations

import pandas as pd

POP_COLUMNS = ["pop_buy_log", "pop_view_log", "pop_buy_rank_pct", "pop_ctr"]
DEFAULT_KEY = "__default__"
SHOWN_EXPRS = [
    "CAST(shown, 'Array(String)')",
    "CAST(shown, 'Array(Nullable(String))')",
    "dynamicElement(shown, 'Array(Nullable(String))')",
    "dynamicElement(shown, 'Array(String)')",
]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)   # RawTree returns counts as strings


def load_popularity(query_fn, until: str | None = None, use_ctr: bool = True, smooth: float = 50.0) -> dict:
    """until: 'YYYY-MM-DD' -> only events/sessions with ts before it.
    pop_buy_log / pop_view_log : log1p of purchases / views in history_events
    pop_buy_rank_pct           : percentile of purchases among products (1 = best seller)
    pop_ctr                    : clicks / impressions from history_sessions, shrunk to the global rate (smooth = pseudo-impressions)
    """
    import numpy as np

    where_ev = f"WHERE ts::String < '{until}'" if until else ""
    where_ss = f"WHERE ts::String < '{until}'" if until else ""

    ev = query_fn(f"""
        SELECT product_id::String AS product_id,
               countIf(event_type::String = 'view') AS n_view,
               countIf(event_type::String = 'purchase') AS n_buy
        FROM shop.history_events {where_ev}
        GROUP BY product_id""")
    ev = pd.DataFrame(ev) if len(ev) else pd.DataFrame(columns=["product_id", "n_view", "n_buy"])

    ss = pd.DataFrame(columns=["product_id", "imps", "clicks"])
    if use_ctr:
        # `shown` and `clicked` are Dynamic columns: arrayJoin needs a typed array, so cast first. JSON arrays are
        # usually stored as Array(Nullable(String)); we try several casts and use the first one the server accepts.
        # `clicked` may be one id or a list: position() on its string form covers both.
        for expr in SHOWN_EXPRS:
            try:
                got = query_fn(f"""
                    SELECT p AS product_id, count() AS imps, countIf(position(toString(c), p) > 0) AS clicks
                    FROM (SELECT arrayJoin({expr}) AS p, clicked AS c FROM shop.history_sessions {where_ss})
                    WHERE isNotNull(p)
                    GROUP BY p""")
                ss = pd.DataFrame(got) if len(got) else ss
                print(f"[popularity] CTR from history_sessions using: {expr}", flush=True)
                break
            except Exception as e:
                body = getattr(getattr(e, "response", None), "text", "")[:300].replace("\n", " ")
                print(f"[popularity] CTR attempt failed ({expr}): {body or e}", flush=True)
        else:
            print("[popularity] all CTR attempts failed: pop_ctr falls back to the global rate", flush=True)

    ev = ev.assign(n_view=_num(ev, "n_view"), n_buy=_num(ev, "n_buy")) if len(ev) else ev
    ss = ss.assign(imps=_num(ss, "imps"), clicks=_num(ss, "clicks")) if len(ss) else ss
    df = ev.merge(ss, on="product_id", how="outer").fillna({"n_view": 0.0, "n_buy": 0.0, "imps": 0.0, "clicks": 0.0})

    prior = float(ss["clicks"].sum() / ss["imps"].sum()) if len(ss) and ss["imps"].sum() > 0 else 0.0
    default = {"pop_buy_log": 0.0, "pop_view_log": 0.0, "pop_buy_rank_pct": 0.0, "pop_ctr": prior}
    pop = {DEFAULT_KEY: default}
    if df.empty:
        return pop

    df["pop_buy_log"] = np.log1p(df["n_buy"])
    df["pop_view_log"] = np.log1p(df["n_view"])
    df["pop_buy_rank_pct"] = df["n_buy"].rank(pct=True)
    df["pop_ctr"] = (df["clicks"] + smooth * prior) / (df["imps"] + smooth)
    for pid, row in zip(df["product_id"], df[POP_COLUMNS].to_dict("records")):
        pop[str(pid)] = {k: float(v) for k, v in row.items()}
    return pop