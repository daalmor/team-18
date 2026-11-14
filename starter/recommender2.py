"""LightGBM recommender v3: poller -> worker pool. Newest first, skips requests already too old to count.

    python starter/recommender_v3.py            # live loop
    python starter/recommender_v3.py --check    # compares build_frame vs the old per-row path, no push
"""
from __future__ import annotations

import os
import pickle
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

import userFeatures
from coviewFeatures import extract_coview_features_from_events
from featureConstructor import FeatureConstructor, _cross_features, _last_viewed_id

# ---------------- knobs ----------------
WORKERS = 6              # parallel answers (network + pandas; LightGBM predict runs with 1 thread each)
BUDGET_S = 2.25          # max seconds building features + scoring ONE request (then: co-view fallback)
START_MAX_AGE_S = 2.4    # don't start a request older than this: it would miss the shop's deadline (seen >= 3.4 s)
N_CV, N_TOTAL = 2000, 500   # candidates per request: co-views first, padded with best sellers
POLL_SLEEP_S = 0.05
LOG_EVERY_S = 2
PUSH_RETRIES = 1

for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ["TEAM_ID"]
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"
_adapter = HTTPAdapter(pool_connections=4, pool_maxsize=WORKERS * 2 + 4)   # reuse connections across threads
http.mount("https://", _adapter)
http.mount("http://", _adapter)

Record = dict[str, Any]

SALES = """SELECT product_id, count() AS n FROM shop.history_events
WHERE event_type = 'purchase'{sales} GROUP BY product_id"""
NEW_REQUESTS = """SELECT request_id, ts, user_id, context, recent_views
FROM shop.requests ORDER BY ts DESC LIMIT 30"""
OUT_OF_STOCK = """SELECT product_id::String AS product_id FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id
HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'"""
COVIEWS_SQL = """
SELECT a.product_id AS p, b.product_id AS q, count() AS n
FROM (SELECT session_id::String AS session_id, product_id::String AS product_id FROM shop.history_events WHERE event_type = 'view') AS a
INNER JOIN (SELECT session_id::String AS session_id, product_id::String AS product_id FROM shop.history_events WHERE event_type = 'view') AS b
  ON a.session_id = b.session_id
WHERE a.product_id != b.product_id
GROUP BY p, q ORDER BY n DESC LIMIT 20 BY p"""


def query(sql: str, timeout: int = 10) -> list[Record]:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
    r.raise_for_status()
    return r.json()["data"]


def query_df(sql: str) -> pd.DataFrame:           # what FeatureConstructor / userFeatures expect
    return pd.DataFrame(query(sql, timeout=60))


# ---------- one-time setup (heavy stuff happens at start, not per request) ----------
t0 = time.time()
print(f"[startup] Starting recommender for team '{TEAM}' -> {API}", flush=True)

print("[startup] Loading LightGBM model and schema...", flush=True)
model = lgb.Booster(model_file="starter/model.txt")
schema = pickle.load(open("starter/schema.pkl", "rb"))
print(f"[startup] Model & schema loaded ({time.time() - t0:.1f}s)", flush=True)

t_step = time.time()
print("[startup] Fetching shop.catalog...", flush=True)
catalog = query_df("SELECT * FROM shop.catalog")
print(f"[startup] Catalog fetched: {len(catalog)} products ({time.time() - t_step:.1f}s)", flush=True)

t_step = time.time()
print("[startup] Running COVIEWS_SQL (heavy self-join, may take a few minutes)...", flush=True)
_cv = pd.DataFrame(query(COVIEWS_SQL, timeout=600))
COVIEWS = _cv.groupby("p").q.apply(list).to_dict() if not _cv.empty else {}
print(f"[startup] Coviews built for {len(COVIEWS)} products ({time.time() - t_step:.1f}s)", flush=True)

# Same flags as in training (USE_IALS=False in the notebook)
print("[startup] Initializing FeatureConstructor...", flush=True)
fc = FeatureConstructor(query_function=query_df, catalog=catalog, coview_dict=COVIEWS,
                        use_ials_features=False)
fc.feature_columns = schema["feature_columns"]
fc.column_dtypes = schema["column_dtypes"]
fc.category_dtypes = schema["category_dtypes"]

CREATED = pd.to_datetime(catalog.set_index("product_id")["created_at"], errors="coerce").to_dict()
print(f"[startup] Setup complete in {time.time() - t0:.1f}s total.", flush=True)


# ---------- stats ----------
class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.t: dict[str, float] = defaultdict(float)   # accumulated seconds per stage
        self.n = 0                                        # answered (pushed)
        self.dropped = 0                                  # skipped: too old to count
        self.push_failed = 0
        self.versions: Counter = Counter()
        self.age_sum = self.e2e_sum = self.e2e_max = 0.0
        self.batches = 0

    def add_time(self, key: str, seconds: float) -> None:
        with self.lock:
            self.t[key] += seconds

    def log(self) -> None:
        with self.lock:
            n = self.n
            if n == 0 and self.dropped == 0:
                return
            per_req = " ".join(f"{k}={1000 * v / max(n, 1):.0f}ms" for k, v in self.t.items() if k not in ("prefetch", "user_feat"))
            per_batch = f"prefetch={1000 * self.t.get('prefetch', 0) / max(self.batches, 1):.0f}ms/batch"
            user_feat = f"user_feat={1000 * self.t.get('user_feat', 0) / max(n + self.dropped, 1):.0f}ms/req"
            print(f"[timing] answered={n} dropped_stale={self.dropped} push_failed={self.push_failed} {dict(self.versions)} | "
                  f"age_at_start avg={self.age_sum / max(n, 1):.2f}s | e2e avg={self.e2e_sum / max(n, 1):.2f}s max={self.e2e_max:.2f}s | "
                  f"{per_req} | {per_batch} {user_feat}", flush=True)
            self.reset()


STATS = Stats()


# ---------- data helpers ----------
def best_sellers() -> list[str]:
    t_step = time.time()
    print("[init] Fetching best_sellers()...", flush=True)
    rows = query(SALES.format(sales=""), timeout=600)
    ranked = [r["product_id"] for r in sorted(rows, key=lambda r: (int(r["n"]), r["product_id"]), reverse=True)]
    print(f"[init] Loaded {len(ranked)} best sellers ({time.time() - t_step:.1f}s)", flush=True)
    return ranked


def out_of_stock() -> set[str]:
    return {r["product_id"] for r in query(OUT_OF_STOCK)}


def pull(seen: set[str]) -> list[Record]:
    fresh = []
    for req in query(NEW_REQUESTS):                 # newest first
        if req["request_id"] not in seen:
            seen.add(req["request_id"])
            fresh.append(req)
    return fresh


def _parse_ts(ts_raw) -> pd.Timestamp:
    ts = pd.to_datetime(ts_raw, errors="coerce")
    if pd.notna(ts) and ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def req_age(req: Record) -> float:
    """Seconds since the request was created (our clock vs the shop's UTC timestamp)."""
    ts = _parse_ts(req.get("ts"))
    return (pd.Timestamp.now(tz="UTC").tz_localize(None) - ts).total_seconds() if pd.notna(ts) else 0.0


def prefetch_users(reqs: list[Record]) -> None:
    """Download the history of every user in this poll at once (2 queries per poll, not 2 per request)."""
    uids = list({r["user_id"] for r in reqs if r.get("user_id")})
    if not uids:
        return
    ts_max = pd.to_datetime([r.get("ts") for r in reqs], errors="coerce").max()
    try:
        userFeatures.prefetch(uids, ts_max, show_progress=False)
    except Exception as e:
        print(f"[warn] prefetch failed (falling back to per-user queries): {e}", flush=True)


# ---------- candidates + features ----------
def candidates(recent_views, best, unavailable, n_cv=N_CV, n_total=N_TOTAL) -> list[str]:
    """Co-views of the recent views (most recent weighted higher), padded with best sellers."""
    score = Counter()
    for i, v in enumerate(reversed(recent_views)):
        pid = v["product_id"] if isinstance(v, dict) else v
        for r, q in enumerate(COVIEWS.get(pid, [])):
            score[q] += 1.0 / ((1 + i) * (1 + r))
    cands = [p for p, _ in score.most_common() if p not in unavailable][:n_cv]
    have = set(cands)
    for p in best:
        if len(cands) >= n_total:
            break
        if p not in unavailable and p not in have:
            cands.append(p)
            have.add(p)
    return cands


def _user_data(req: Record, ts_raw, ts) -> dict:
    """User features, computed ONCE per request. Anonymous users get the same defaults as in training."""
    uid = req.get("user_id")
    if uid:
        return userFeatures.user_features(uid, ts_raw) or {}
    no_hist = pd.DataFrame({c: [] for c in ["session_id", "event_type", "product_id", "ts"]})
    no_sess = pd.DataFrame({c: [] for c in ["request_id", "ts", "context", "clicked", "purchased"]})
    when = ts if pd.notna(ts) else pd.Timestamp.now(tz="UTC").tz_localize(None)
    d = dict(userFeatures.build_user_features(no_hist, no_sess, when))
    d["is_new_user"] = 1
    return d


def safe_user_data(req: Record) -> dict | None:
    """Run in the poller thread (right after prefetch) so workers never touch userFeatures' shared cache."""
    try:
        ts_raw = req.get("ts")
        return _user_data(req, ts_raw, _parse_ts(ts_raw))
    except Exception as e:
        print(f"[warn] user features failed for {req.get('request_id')}: {e}", flush=True)
        return None


def build_frame(cands: list[str], req: Record, deadline: float, user_data: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """One DataFrame for all candidates of a request (mirrors the inner loop of generate_xy).
    User and session features are computed once; only item-specific ones per candidate.
    Raises TimeoutError if `deadline` (epoch seconds) is passed."""
    ctx = req.get("context") or {}
    rv = req.get("recent_views") or []
    ts_raw = req.get("ts")
    ts = _parse_ts(ts_raw)
    hour = int(ts.hour) if pd.notna(ts) else -1
    dow = int(ts.day_of_week) if pd.notna(ts) else -1

    if user_data is None:
        user_data = _user_data(req, ts_raw, ts)
    if time.time() > deadline:
        raise TimeoutError("budget exceeded after user features")
    last_meta = fc.catalog_lookup.get(_last_viewed_id(rv)) if rv else None

    rows, kept = [], []
    for n, p in enumerate(cands):
        if n % 10 == 0 and time.time() > deadline:
            raise TimeoutError(f"budget exceeded after {n} candidates")
        m = fc.catalog_lookup.get(p)
        if m is None:
            continue
        price = float(m.get("price_eur") or 0.0)
        created = CREATED.get(p)
        X = {
            "brand": m.get("brand"), "category": m.get("category"), "subcategory": m.get("subcategory"),
            "device": ctx.get("device"), "region": ctx.get("region"), "page": ctx.get("page"),
            "price_eur": price,
            "product_age_days": max((ts - created).total_seconds() / 86400.0, 0.0)
                                if pd.notna(ts) and pd.notna(created) else -1.0,
            "description_len": len(m.get("description") or ""),
            "is_new_user": int(user_data.get("is_new_user", 0)),
            "position": 0, "hour_of_day": hour, "day_of_week": dow, "is_weekend": int(dow in (5, 6)),
            "session_length": len(rv),
            "same_category_as_last_view": int(bool(last_meta) and m.get("category") == last_meta.get("category")),
            "same_brand_as_last_view": int(bool(last_meta) and m.get("brand") == last_meta.get("brand")),
            "price_diff_from_last_view": (price - float(last_meta["price_eur"])
                                          if last_meta and last_meta.get("price_eur") is not None else 0.0),
        }
        X.update(extract_coview_features_from_events(p, rv, COVIEWS))
        X.update(user_data)
        X.update(_cross_features(user_data, m.get("category"), price))
        rows.append(X)
        kept.append(p)

    df = pd.DataFrame(rows).reindex(columns=schema["feature_columns"])
    for c, dt in schema["column_dtypes"].items():
        if c in df.columns:
            try:
                df[c] = df[c].astype(dt)
            except (ValueError, TypeError):
                pass
    for c, dt in schema["category_dtypes"].items():
        if c in df.columns:
            df[c] = df[c].astype(dt)
    return df, kept


# ---------- answering (runs in worker threads) ----------
def answer(req: Record, best: list[str], unavailable: set[str], user_data: dict | None) -> Record:
    t_start = time.time()
    deadline = t_start + BUDGET_S
    cands = candidates(req.get("recent_views") or [], best, unavailable)
    STATS.add_time("cands", time.time() - t_start)
    try:
        t = time.time()
        X, kept = build_frame(cands, req, deadline, user_data)
        STATS.add_time("frame", time.time() - t)
        t = time.time()
        scores = model.predict(X, num_threads=1)     # 1 thread: several workers predict at once
        STATS.add_time("model", time.time() - t)
        items = [kept[i] for i in np.argsort(-scores)[:10]]
        items += [p for p in cands if p not in items][: 10 - len(items)]   # safety pad
        version = "lgbm_v1"
    except Exception as e:                           # never leave a request unanswered
        reason = "time budget" if isinstance(e, TimeoutError) else repr(e)
        print(f"[warn] fallback for {req['request_id']}: {reason}", flush=True)
        # co-view candidates are already ranked and in stock: better than the plain baseline
        items = cands[:10] if len(cands) >= 10 else [p for p in best if p not in unavailable][:10]
        version = "fallback_cv"
    return {"request_id": req["request_id"], "items": items, "model_version": version}


def push(answers: list[Record]) -> bool:
    """Insert into team_NN.submissions. Returns False if it never went through."""
    for attempt in range(PUSH_RETRIES + 1):
        try:
            http.post(f"{API}/v1/tables/submissions", params={"database": TEAM},
                      json=answers, timeout=10).raise_for_status()
            return True
        except Exception as e:
            print(f"[warn] push failed (attempt {attempt + 1}): {e}", flush=True)
    return False


def handle(req: Record, user_data: dict | None, best: list[str], state: dict) -> None:
    try:
        age0 = req_age(req)
        if age0 > START_MAX_AGE_S:                   # already too old to count: don't waste a worker on it
            with STATS.lock:
                STATS.dropped += 1
            return
        ans = answer(req, best, state["unavailable"], user_data)
        ok = push([ans])
        e2e = req_age(req)
        with STATS.lock:
            STATS.n += 1
            STATS.versions[ans["model_version"]] += 1
            STATS.age_sum += age0
            STATS.e2e_sum += e2e
            STATS.e2e_max = max(STATS.e2e_max, e2e)
            if not ok:
                STATS.push_failed += 1
    except Exception as e:
        print(f"[error] handle {req.get('request_id')}: {e}", flush=True)


# ---------- main loop ----------
def stock_loop(state: dict) -> None:
    """Refresh the out-of-stock set in the background so it never delays polling."""
    while True:
        try:
            state["unavailable"] = out_of_stock()
        except Exception as e:                       # keep answering with the last known stock
            print(f"[warn] stock check failed: {e}", flush=True)
        time.sleep(1)


def main() -> None:
    best: list[str] = []
    while not best:
        try:
            best = best_sellers()
        except Exception as e:
            print(f"[error] best_sellers: {e}", flush=True)
            time.sleep(2)

    state: dict[str, Any] = {"unavailable": set()}
    try:
        state["unavailable"] = out_of_stock()
    except Exception as e:
        print(f"[warn] initial stock check failed: {e}", flush=True)
    threading.Thread(target=stock_loop, args=(state,), daemon=True).start()

    pool = ThreadPoolExecutor(max_workers=WORKERS)
    seen = set[str]()
    last_log = last_heartbeat = time.time()

    print(f"[loop] Entering main polling loop ({WORKERS} workers, skip requests older than {START_MAX_AGE_S}s)...", flush=True)
    while True:
        try:
            fresh = pull(seen)
            if fresh:
                t = time.time()
                prefetch_users(fresh)
                STATS.add_time("prefetch", time.time() - t)
                with STATS.lock:
                    STATS.batches += 1
                for req in fresh:                    # newest first
                    t = time.time()
                    user_data = safe_user_data(req)
                    STATS.add_time("user_feat", time.time() - t)
                    pool.submit(handle, req, user_data, best, state)
            if time.time() - last_log >= LOG_EVERY_S:
                STATS.log()
                last_log = time.time()
            elif not fresh and time.time() - last_heartbeat >= 15:
                print(f"[heartbeat] polling... (seen={len(seen)}, out_of_stock={len(state['unavailable'])})", flush=True)
                last_heartbeat = time.time()
        except Exception as e:  # keep running: shoppers keep arriving while you sleep
            print(f"[error] {e}", flush=True)
            time.sleep(1)
        time.sleep(POLL_SLEEP_S)


# ---------- offline sanity check ----------
def check(n_req: int = 3, n_cand: int = 15) -> None:
    """build_frame must give the same features and scores as the old per-row path (no push)."""
    best, unavailable = best_sellers(), out_of_stock()
    reqs = pull(set())[:n_req]
    prefetch_users(reqs)
    for req in reqs:
        ctx = req.get("context") or {}
        cands = candidates(req.get("recent_views") or [], best, unavailable)[:n_cand]
        X_new, kept = build_frame(cands, req, time.time() + 60)
        X_old = pd.concat([fc.build_prediction_row(
            product_id=p, user_id=req.get("user_id"), ts=req.get("ts"), recent_views=req.get("recent_views") or [],
            device=ctx.get("device"), region=ctx.get("region"), page=ctx.get("page"),
            is_new_user=req.get("user_id") is None, position=0) for p in kept], ignore_index=True)
        bad = []
        for c in X_old.columns:
            a, b = X_new[c].reset_index(drop=True), X_old[c].reset_index(drop=True)
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                ok = np.isclose(a.astype(float), b.astype(float), equal_nan=True).all()
            else:
                a, b = a.astype(object), b.astype(object)
                ok = ((a == b) | (a.isna() & b.isna())).all()
            if not ok:
                bad.append(c)
        s_new, s_old = model.predict(X_new), model.predict(X_old)
        same_top = list(np.argsort(-s_new)[:10]) == list(np.argsort(-s_old)[:10])
        print(f"[check] {req['request_id']}: mismatching columns={bad or 'none'} | "
              f"scores close={np.allclose(s_new, s_old)} | same top-10={same_top}", flush=True)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        main()