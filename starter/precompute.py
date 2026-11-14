"""Precompute the whole user-history cache once (history is static) and save it to disk.

    python starter/precompute.py                  # all users (resumes if starter/history_cache.pkl already exists)
    python starter/precompute.py --limit 20000    # only the first 20000 users (recent requesters first)

recommender_v4.py loads starter/history_cache.pkl at startup, so no known user is ever cold and restarts take seconds.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import requests

import userFeatures

for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"

CACHE_PATH = "starter/history_cache.pkl"
CHUNK = 2000                              # users per prefetch: each chunk = 2 table scans, so bigger = fewer scans
SAVE_EVERY = 5                            # chunks
FAR_FUTURE = pd.Timestamp("2100-01-01")   # include ALL history events


def query(sql: str, timeout: int = 600) -> list[dict]:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
    r.raise_for_status()
    return r.json()["data"]


def query_df(sql: str) -> pd.DataFrame:
    return pd.DataFrame(query(sql))


def save() -> None:
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(userFeatures._HIST, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, CACHE_PATH)


def main() -> None:
    t0 = time.time()
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    userFeatures.init(query_df, query_df("SELECT * FROM shop.catalog"))
    if Path(CACHE_PATH).exists():
        userFeatures._HIST.update(pickle.load(open(CACHE_PATH, "rb")))
        print(f"[precompute] resuming: {len(userFeatures._HIST['users'])} users already cached", flush=True)

    print("[precompute] listing users...", flush=True)
    recent = [r["user_id"] for r in query(
        "SELECT user_id::String AS user_id FROM shop.requests GROUP BY user_id ORDER BY max(ts::String) DESC")]
    in_sessions = [r["user_id"] for r in query(
        "SELECT user_id::String AS user_id FROM shop.history_sessions GROUP BY user_id")]
    in_events = [r["user_id"] for r in query(
        "SELECT user_id::String AS user_id FROM shop.history_events GROUP BY user_id")]
    users = list(dict.fromkeys(recent + in_sessions + in_events))        # recent requesters first
    print(f"[precompute] users: {len(recent)} in requests, {len(in_sessions)} in sessions, "
          f"{len(in_events)} in events -> {len(users)} distinct ({time.time() - t0:.0f}s)", flush=True)
    if limit:
        users = users[:limit]
    todo = [u for u in users if u not in userFeatures._HIST["users"]]
    print(f"[precompute] to download: {len(todo)} users in chunks of {CHUNK}", flush=True)

    t_dl = time.time()
    for n, i in enumerate(range(0, len(todo), CHUNK), start=1):
        chunk = todo[i:i + CHUNK]
        userFeatures.prefetch(chunk, FAR_FUTURE, chunk=CHUNK, show_progress=False)
        done = min(i + CHUNK, len(todo))
        elapsed = time.time() - t_dl
        eta = elapsed / done * (len(todo) - done)
        print(f"[precompute] {done}/{len(todo)} users | {elapsed:.0f}s elapsed | ETA {eta:.0f}s", flush=True)
        if n % SAVE_EVERY == 0:
            save()
    save()
    size_mb = Path(CACHE_PATH).stat().st_size / 1e6
    print(f"[precompute] done: {len(userFeatures._HIST['users'])} users cached -> {CACHE_PATH} "
          f"({size_mb:.0f} MB, {time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()