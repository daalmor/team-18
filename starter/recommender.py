"""LightGBM recommender v4: poller -> workers. User history comes from an in-memory cache that is warmed in the
background; a cold user is answered right away with new-user defaults (and fetched for next time).

    python starter/recommender_v4.py            # live loop
    python starter/recommender_v4.py --check    # compares build_frame vs the old per-row path, no push
"""
from __future__ import annotations

import functools
import os
import pickle
import queue
import sys
import threading
import time
import zlib
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
from popularity import load_popularity

# ---------------- knobs ----------------
WORKERS = 6              # parallel answers (LightGBM predict runs with 1 thread each)
BUDGET_S = 2.25          # max seconds for user features + frame + scoring of ONE request (then: co-view fallback)
START_MAX_AGE_S = 2.5    # don't start a request older than this: it would miss the shop's deadline (seen >= 3.4 s)
N_CV, N_TOTAL = 40, 80   # candidates per request: up to 40 co-views, then the best in-stock sellers up to 80 in total
POLL_SLEEP_S = 0.05
LOG_EVERY_S = 15
PUSH_RETRIES = 1
WARM_USERS = 5000        # history cache warm-up: most recent distinct users in shop.requests
WARM_CHUNK = 500
WARM_MAX_S = 240
BUY_WEIGHT = 10.0        # rank by p_click + BUY_WEIGHT * p_buy (offline: best top-3 for both clicks and buys at 10)
# Users WITHOUT history (new, anonymous or not cached yet) are split between these strategies by request_id, so the
# feedback table tells us which works best: "lgbm" = the model with no-history features, "cv" = co-view order
# (then best sellers), "pop" = best sellers. Use ("lgbm",) to switch the test off.
NOHIST_ARMS = ("lgbm", "cv", "pop")

# Strategy for ALL requests, chosen on the command line:  python recommender_v4.py --mode cv
#   lgbm  (default) the model; users without history go through NOHIST_ARMS
#   cv    co-view order for everyone: no model, no user history, very fast (label "cv_all")
#   split 50/50 by request_id between the model and co-view order, for EVERY user: a fair live comparison
#         (labels "ab_model" vs "ab_cv": the leaderboard rows compare them directly)
MODE = "lgbm"
for _i, _a in enumerate(sys.argv):
    if _a == "--mode" and _i + 1 < len(sys.argv):
        MODE = sys.argv[_i + 1]
if MODE not in ("lgbm", "cv", "split"):
    raise SystemExit(f"--mode must be lgbm, cv or split (got {MODE!r})")

for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ["TEAM_ID"]
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"
_adapter = HTTPAdapter(pool_connections=4, pool_maxsize=WORKERS * 2 + 6)   # reuse connections across threads
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
WARM_USERS_SQL = """SELECT user_id::String AS user_id, max(ts::String) AS last_ts
FROM shop.requests GROUP BY user_id ORDER BY last_ts DESC LIMIT {n}"""
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
MODEL_BUY_PATH = "starter/model_buy.txt"
model_buy = lgb.Booster(model_file=MODEL_BUY_PATH) if Path(MODEL_BUY_PATH).exists() else None
print(f"[startup] Purchase model: {'loaded (blend weight %.1f)' % BUY_WEIGHT if model_buy else 'NOT found, ranking by clicks only'}", flush=True)
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

# Popularity features (only if the model was trained with them): precomputed by the notebook, else computed now.
POP_PATH = "starter/popularity.pkl"
POP_COLS_IN_MODEL = [c for c in schema["feature_columns"] if c.startswith("pop_")]
if POP_COLS_IN_MODEL:
    t_step = time.time()
    if Path(POP_PATH).exists():
        fc.pop_lookup = pickle.load(open(POP_PATH, "rb"))
    else:
        print("[startup] No popularity.pkl: computing popularity from RawTree (may take a minute)...", flush=True)
        fc.pop_lookup = load_popularity(lambda sql: pd.DataFrame(query(sql, timeout=600)))
    missing_pop = [c for c in POP_COLS_IN_MODEL if c not in fc._pop(None)]
    if missing_pop:
        raise RuntimeError(f"model expects popularity columns we cannot compute: {missing_pop}")
    print(f"[startup] Popularity loaded: {len(fc.pop_lookup) - 1} products ({time.time() - t_step:.1f}s)", flush=True)

# History is static, so the whole user-history cache is precomputed once (python starter/precompute.py).
HISTORY_CACHE = "starter/history_cache.pkl"
if Path(HISTORY_CACHE).exists():
    t_step = time.time()
    userFeatures._HIST.update(pickle.load(open(HISTORY_CACHE, "rb")))
    print(f"[startup] History cache loaded: {len(userFeatures._HIST['users'])} users ({time.time() - t_step:.1f}s)", flush=True)
else:
    print("[startup] No history cache found (run precompute.py once): users are fetched in the background", flush=True)

# The cache is never refreshed per user, so mark it valid for any request time.
# Otherwise fetch_user_history() runs 2 slow RawTree queries whenever a request is newer than the last prefetch.
userFeatures._HIST["t_max"] = pd.Timestamp("2100-01-01")
print(f"[startup] Setup complete in {time.time() - t0:.1f}s total.", flush=True)


# ---------- stats ----------
class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.t: dict[str, float] = defaultdict(float)   # accumulated seconds per stage (per answered request)
        self.n = 0                                        # answered (pushed)
        self.dropped = 0                                  # skipped: too old to count
        self.push_failed = 0
        self.versions: Counter = Counter()
        self.age_sum = self.e2e_sum = self.e2e_max = 0.0
        self.hist_n = 0                                   # background history fetches
        self.hist_s = 0.0

    def add_time(self, key: str, seconds: float) -> None:
        with self.lock:
            self.t[key] += seconds

    def log(self) -> None:
        with self.lock:
            n = self.n
            if n == 0 and self.dropped == 0:
                return
            per_req = " ".join(f"{k}={1000 * v / max(n, 1):.0f}ms" for k, v in self.t.items())
            hist = f"hist_fetches={self.hist_n} (avg {self.hist_s / max(self.hist_n, 1):.1f}s) pending={len(PENDING)}"
            print(f"[timing] answered={n} dropped_stale={self.dropped} push_failed={self.push_failed} {dict(self.versions)} | "
                  f"age_at_start avg={self.age_sum / max(n, 1):.2f}s | e2e avg={self.e2e_sum / max(n, 1):.2f}s "
                  f"max={self.e2e_max:.2f}s | {per_req} | {hist}", flush=True)
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


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def _parse_ts(ts_raw) -> pd.Timestamp:
    ts = pd.to_datetime(ts_raw, errors="coerce")
    if pd.notna(ts) and ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def req_age(req: Record) -> float:
    """Seconds since the request was created (our clock vs the shop's UTC timestamp)."""
    ts = _parse_ts(req.get("ts"))
    return (_now_utc() - ts).total_seconds() if pd.notna(ts) else 0.0


# ---------- user history cache (userFeatures._HIST) ----------
PENDING: set[str] = set()                 # users already queued for a background fetch
PENDING_LOCK = threading.Lock()
HISTORY_Q: queue.Queue = queue.Queue()


def request_history(uid, ts) -> None:
    """Queue a user for a background history fetch (deduplicated). Their NEXT request will be warm."""
    uid = str(uid)
    with PENDING_LOCK:
        if uid in PENDING:
            return
        PENDING.add(uid)
    HISTORY_Q.put((uid, ts))


def history_loop() -> None:
    """Collects cold users for a moment and fetches them together: 2 queries per group, not 2 per user."""
    while True:
        uid, ts = HISTORY_Q.get()
        batch = {uid: ts}
        time.sleep(0.2)
        while True:
            try:
                u, t = HISTORY_Q.get_nowait()
                batch[u] = t
            except queue.Empty:
                break
        valid = [t for t in batch.values() if pd.notna(t)]
        ts_max = max(valid) if valid else _now_utc()
        t_start = time.time()
        try:
            userFeatures.prefetch(list(batch), ts_max, chunk=WARM_CHUNK, show_progress=False)
        except Exception as e:
            print(f"[warn] history fetch failed for {len(batch)} users: {e}", flush=True)
        finally:
            with PENDING_LOCK:
                PENDING.difference_update(batch)
            with STATS.lock:
                STATS.hist_n += 1
                STATS.hist_s += time.time() - t_start


def warm_loop() -> None:
    """Fill the history cache for the most recent distinct users, in the background."""
    t_start = time.time()
    try:
        rows = query(WARM_USERS_SQL.format(n=WARM_USERS), timeout=120)
    except Exception as e:
        print(f"[warn] warm-up query failed: {e}", flush=True)
        return
    uids = [r["user_id"] for r in rows if r.get("user_id")]
    if not uids:
        return
    ts_max = _parse_ts(max(r["last_ts"] for r in rows))
    done = 0
    for i in range(0, len(uids), WARM_CHUNK):
        if time.time() - t_start > WARM_MAX_S:
            break
        chunk = uids[i:i + WARM_CHUNK]
        try:
            userFeatures.prefetch(chunk, ts_max, chunk=WARM_CHUNK, show_progress=False)
            done += len(chunk)
        except Exception as e:
            print(f"[warn] warm-up chunk failed: {e}", flush=True)
    print(f"[warm] cached {done}/{len(uids)} users in {time.time() - t_start:.0f}s", flush=True)


@functools.lru_cache(maxsize=512)
def _defaults_for_hour(hour_ts: pd.Timestamp) -> dict:
    """User features of someone with no history (only the time-of-day features vary), cached per hour."""
    no_hist = pd.DataFrame({c: [] for c in ["session_id", "event_type", "product_id", "ts"]})
    no_sess = pd.DataFrame({c: [] for c in ["request_id", "ts", "context", "clicked", "purchased"]})
    d = dict(userFeatures.build_user_features(no_hist, no_sess, hour_ts))
    d["is_new_user"] = 1
    return d


def default_user_data(ts) -> dict:
    when = ts if pd.notna(ts) else _now_utc()
    return dict(_defaults_for_hour(when.floor("h")))


def user_data_for(req: Record) -> tuple[dict, bool]:
    """(user features, warm). Never touches the network: a cold user gets new-user defaults and is queued."""
    uid = req.get("user_id")
    ts_raw = req.get("ts")
    ts = _parse_ts(ts_raw)
    if not uid:
        return default_user_data(ts), False
    if str(uid) in userFeatures._HIST["users"]:
        return userFeatures.user_features(uid, ts_raw) or {}, True
    request_history(uid, ts)
    return default_user_data(ts), False


def prefetch_users(reqs: list[Record]) -> None:
    """Blocking prefetch for a list of requests (used by --check)."""
    uids = list({r["user_id"] for r in reqs if r.get("user_id")})
    if uids:
        ts_max = pd.to_datetime([r.get("ts") for r in reqs], errors="coerce").max()
        userFeatures.prefetch(uids, ts_max, show_progress=False)


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

    if user_data is None:                            # --check path: real history (call prefetch_users first)
        uid = req.get("user_id")
        user_data = (userFeatures.user_features(uid, ts_raw) or {}) if uid else default_user_data(ts)
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
        if POP_COLS_IN_MODEL:
            X.update(fc._pop(p))
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
            df[c] = pd.Categorical(df[c], categories=dt.categories)
    return df, kept


# ---------- answering (runs in worker threads) ----------
def _arm(req: Record) -> str:
    return NOHIST_ARMS[zlib.crc32(str(req["request_id"]).encode()) % len(NOHIST_ARMS)]


def answer(req: Record, best: list[str], unavailable: set[str]) -> Record:
    t_start = time.time()
    deadline = t_start + BUDGET_S
    cands = candidates(req.get("recent_views") or [], best, unavailable)
    STATS.add_time("cands", time.time() - t_start)
    if MODE == "cv" or (MODE == "split" and zlib.crc32(("ab:" + str(req["request_id"])).encode()) % 2 == 1):
        items = cands[:10]                           # candidates() ranks co-views first, then best sellers
        if len(items) < 10:
            items += [p for p in best if p not in unavailable and p not in items][: 10 - len(items)]
        return {"request_id": req["request_id"], "items": items,
                "model_version": "cv_all" if MODE == "cv" else "ab_cv"}
    try:
        t = time.time()
        user_data, warm = user_data_for(req)
        STATS.add_time("user_feat", time.time() - t)
        if MODE == "split":
            version = "ab_model"                     # the model for everyone, with or without history
        elif warm and float(user_data.get("is_new_user", 0)) < 0.5:
            version = "lgbm_v1"                      # user with real history
        else:
            arm = _arm(req)
            version = f"nohist_{arm}"
            if arm == "cv":                          # candidates() already ranks co-views first, then best sellers
                return {"request_id": req["request_id"], "items": cands[:10], "model_version": version}
            if arm == "pop":
                return {"request_id": req["request_id"],
                        "items": [p for p in best if p not in unavailable][:10], "model_version": version}
        t = time.time()
        X, kept = build_frame(cands, req, deadline, user_data)
        STATS.add_time("frame", time.time() - t)
        t = time.time()
        scores = model.predict(X, num_threads=1)     # 1 thread: several workers predict at once
        if model_buy is not None:
            scores = scores + BUY_WEIGHT * model_buy.predict(X, num_threads=1)
        STATS.add_time("model", time.time() - t)
        items = [kept[i] for i in np.argsort(-scores)[:10]]
        items += [p for p in cands if p not in items][: 10 - len(items)]   # safety pad
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


def handle(req: Record, best: list[str], state: dict) -> None:
    try:
        age0 = req_age(req)
        if age0 > START_MAX_AGE_S:                   # already too old to count: don't waste a worker on it
            with STATS.lock:
                STATS.dropped += 1
            return
        ans = answer(req, best, state["unavailable"])
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
    if MODE != "cv":                                 # cv mode never looks at user history
        threading.Thread(target=history_loop, daemon=True).start()
        threading.Thread(target=warm_loop, daemon=True).start()
    print(f"[loop] MODE={MODE}", flush=True)

    pool = ThreadPoolExecutor(max_workers=WORKERS)
    seen = set[str]()
    last_log = last_heartbeat = time.time()

    print(f"[loop] Entering main polling loop ({WORKERS} workers, skip requests older than {START_MAX_AGE_S}s)...", flush=True)
    while True:
        try:
            fresh = pull(seen)
            for req in fresh:                        # newest first
                pool.submit(handle, req, best, state)
            if time.time() - last_log >= LOG_EVERY_S:
                STATS.log()
                last_log = time.time()
            elif not fresh and time.time() - last_heartbeat >= 15:
                print(f"[heartbeat] polling... (seen={len(seen)}, out_of_stock={len(state['unavailable'])}, "
                      f"cached_users={len(userFeatures._HIST['users'])})", flush=True)
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