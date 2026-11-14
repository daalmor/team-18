import json
import re

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# configuracion
EVW = {"view": 1, "add_to_cart": 3, "purchase": 6}   # peso de cada tipo de evento
HALF_LIFE_DAYS = 14                                  # vida media del peso por recencia (gustos)
DEVICES = ["ios", "android", "web"]

_query = None          
_cat = None            
CATEGORIES = []        


def init(query_fn, catalog_df):
    """Hay que llamarla una vez: le pasa a este modulo tu query() y tu catalogo."""
    global _query, _cat, CATEGORIES
    _query = query_fn
    cat = catalog_df.copy()
    cat["product_id"] = cat["product_id"].astype(str)
    cat["price_eur"] = pd.to_numeric(cat["price_eur"], errors="coerce")
    cat["lp"] = np.log1p(cat["price_eur"])                                  
    cat["pct_in_cat"] = cat.groupby("category")["price_eur"].rank(pct=True)  
    _cat = cat[["product_id", "category", "brand", "price_eur", "lp", "pct_in_cat"]]
    CATEGORIES = sorted(cat["category"].dropna().astype(str).unique())


# utilidades
def _ts(x):
    return pd.to_datetime(x, utc=True).tz_localize(None)


def _ctx(c):
    if isinstance(c, dict):
        return c
    if isinstance(c, str):
        try:
            return json.loads(c)
        except Exception:
            return {}
    return {}


def _has(s):
    """True donde clicked / purchased trae un product_id de verdad (los ids son del tipo p_xxxxxx)."""
    return s.astype(str).str.contains(r"p_[0-9a-fA-F]{4,}", regex=True)


# 1) historial del usuario
_HIST = {"ev": {}, "ss": {}, "users": set(), "t_max": None}   # cache en memoria (ver prefetch)
_EV_COLS = ["session_id", "event_type", "product_id", "ts"]
_SS_COLS = ["request_id", "ts", "context", "clicked", "purchased"]


def prefetch(user_ids, ts_max, chunk=300, show_progress=True):
    """Descarga de golpe el historial (anterior a ts_max) de muchos usuarios y lo guarda en memoria.
    Despues, fetch_user_history() lo lee de aqui sin tocar RawTree: 2 consultas por lote de usuarios
    en vez de 2 por peticion."""
    assert _query is not None, "Llama primero a userFeatures.init(query, catalog)"
    ok = lambda x: re.fullmatch(r"[A-Za-z0-9_\-]+", str(x))
    users = sorted({str(x) for x in user_ids if ok(x)} - _HIST["users"])
    t = _ts(ts_max).strftime("%Y-%m-%d %H:%M:%S.%f")
    starts = tqdm(range(0, len(users), chunk), desc=f"Descargando historial ({len(users)} usuarios)",
                  unit="lote", disable=not show_progress)
    for i in starts:
        ids = ",".join(f"'{x}'" for x in users[i:i + chunk])
        ev = _query(f"""
            SELECT user_id::String AS user_id, session_id::String AS session_id, event_type::String AS event_type,
                   product_id::String AS product_id, ts::String AS ts
            FROM shop.history_events WHERE user_id::String IN ({ids}) AND ts::String < '{t}'""")
        ss = _query(f"""
            SELECT user_id::String AS user_id, request_id::String AS request_id, ts::String AS ts,
                   context, clicked, purchased
            FROM shop.history_sessions WHERE user_id::String IN ({ids}) AND ts::String < '{t}'""")
        for df, key in ((ev, "ev"), (ss, "ss")):
            if len(df):
                df["ts"] = pd.to_datetime(df["ts"])
                for uid, g in df.groupby("user_id"):
                    _HIST[key][uid] = g.drop(columns="user_id")
    _HIST["users"].update(users)
    prev = _HIST["t_max"]
    _HIST["t_max"] = _ts(ts_max) if prev is None else max(prev, _ts(ts_max))


def clear_prefetch():
    _HIST.update(ev={}, ss={}, users=set(), t_max=None)


def fetch_user_history(user_id, ts):
    """Eventos y sesiones del usuario anteriores a ts (de la cache si se hizo prefetch, si no de RawTree)."""
    assert _query is not None, "Llama primero a features.init(query, catalog)"
    uid = str(user_id)
    assert re.fullmatch(r"[A-Za-z0-9_\-]+", uid), f"user_id raro: {uid}"
    t = _ts(ts)
    if uid in _HIST["users"] and _HIST["t_max"] is not None and t <= _HIST["t_max"]:
        ev = _HIST["ev"].get(uid, pd.DataFrame(columns=_EV_COLS))
        ss = _HIST["ss"].get(uid, pd.DataFrame(columns=_SS_COLS))
        return ev[ev["ts"] < t], ss[ss["ts"] < t]
    t = t.strftime("%Y-%m-%d %H:%M:%S.%f")
    ev = _query(f"""
        SELECT session_id::String AS session_id, event_type::String AS event_type,
               product_id::String AS product_id, ts::String AS ts
        FROM shop.history_events
        WHERE user_id::String = '{uid}' AND ts::String < '{t}'""")
    ss = _query(f"""
        SELECT request_id::String AS request_id, ts::String AS ts, context, clicked, purchased
        FROM shop.history_sessions
        WHERE user_id::String = '{uid}' AND ts::String < '{t}'""")
    return ev, ss


# 2) historial -> features (funcion pura)
def build_user_features(ev, ss, ts):
    t = _ts(ts)
    ev = ev.copy() if len(ev) else pd.DataFrame({c: [] for c in ["session_id", "event_type", "product_id", "ts"]})
    ev["ts"] = pd.to_datetime(ev["ts"])
    ev = ev.merge(_cat, on="product_id", how="left")
    age = (t - ev["ts"]).dt.total_seconds() / 86400          
    ev["w"] = ev["event_type"].map(EVW).fillna(1)
    v, c, b = (ev["event_type"] == k for k in ("view", "add_to_cart", "purchase"))
    n, nv, nc, nb = len(ev), int(v.sum()), int(c.sum()), int(b.sum())
    f = {}

    # contexto temporal de la peticion (incluye franjas horarias para el Cold Start)
    f["req_hour"], f["req_dow"], f["req_is_weekend"] = t.hour, t.dayofweek, int(t.dayofweek >= 5)
    f["is_morning"] = int(6 <= t.hour < 12)
    f["is_afternoon"] = int(12 <= t.hour < 18)
    f["is_evening"] = int(18 <= t.hour < 24)
    f["is_night"] = int(0 <= t.hour < 6)

    # volumen y embudo (vista -> carrito -> compra)
    f["is_new_user"] = int(n == 0)
    f["n_events"], f["n_views"], f["n_carts"], f["n_buys"] = n, nv, nc, nb
    f["cart_per_view"] = nc / max(nv, 1)
    f["buy_per_view"] = nb / max(nv, 1)
    f["buy_per_cart"] = nb / nc if nc > 0 else 0.0

    # recencia y actividad reciente (NaNs sustituidos por 999.0 para separar ruido de inactividad)
    f["days_since_last_event"] = age.min() if n > 0 else 999.0
    f["minutes_since_last_event"] = age.min() * 1440 if n > 0 else 999.0 * 1440
    f["days_since_first_event"] = age.max() if n > 0 else 999.0
    f["days_since_last_purchase"] = age[b].min() if nb > 0 else 999.0
    f["days_since_last_cart"] = age[c].min() if nc > 0 else 999.0
    for d in (1, 7, 30):
        f[f"events_last_{d}d"] = int((age <= d).sum())
    for d in (7, 30):
        f[f"buys_last_{d}d"] = int(((age <= d) & b).sum())
    f["activity_decayed_7d"] = float((ev["w"] * 0.5 ** (age / 7)).sum())

    # sesiones
    ns = ev["session_id"].nunique()
    f["n_sessions"] = ns
    f["events_per_session"] = n / ns if ns > 0 else 0.0
    st = ev.groupby("session_id")["ts"].min().sort_values()
    f["avg_days_between_sessions"] = (st.diff().dt.total_seconds() / 86400).mean() if len(st) > 1 else 999.0
    f["n_distinct_products"] = ev["product_id"].nunique()
    f["n_distinct_categories"] = ev["category"].nunique()
    f["repeat_view_ratio"] = 1 - ev["product_id"].nunique() / n if n > 0 else 0.0

    # precio habitual (Mantenemos NaNs aquí intencionadamente, los algoritmos de árbol particionan bien los precios nulos sin forzar un "0 euros")
    buys = ev[b]
    f["price_log_mean"], f["price_log_std"], f["price_log_max"] = ev["lp"].mean(), ev["lp"].std(), ev["lp"].max()
    f["price_mean_eur"] = ev["price_eur"].mean()
    f["price_log_mean_last7d"] = ev[age <= 7]["lp"].mean()
    f["buy_price_log_mean"], f["buy_price_mean_eur"] = buys["lp"].mean(), buys["price_eur"].mean()
    f["price_pct_in_cat_mean"] = ev["pct_in_cat"].mean()
    f["buy_price_pct_in_cat_mean"] = buys["pct_in_cat"].mean()

    # gustos: peso por tipo de evento y por recencia
    wr = ev["w"] * 0.5 ** (age / HALF_LIFE_DAYS)
    cs = wr.groupby(ev["category"]).sum()
    cs = cs / cs.sum() if cs.sum() > 0 else cs
    for k in CATEGORIES:
        f[f"cat_share_{k}"] = float(cs.get(k, 0.0))
    f["top_cat_share"] = cs.max() if len(cs) > 0 else 0.0
    f["cat_entropy"] = float(-(cs[cs > 0] * np.log(cs[cs > 0])).sum()) if len(cs) > 0 else 0.0
    bs = wr.groupby(ev["brand"]).sum()
    f["top_brand_share"] = bs.max() / bs.sum() if bs.sum() > 0 else 0.0
    f["n_distinct_brands"] = ev["brand"].nunique()

    # historial del carrusel (lo que ya se le enseñó y como reaccionó)
    ss = ss.copy() if len(ss) else pd.DataFrame({c: [] for c in ["ts", "context", "clicked", "purchased"]})
    ss["ts"] = pd.to_datetime(ss["ts"])
    sage = (t - ss["ts"]).dt.total_seconds() / 86400
    clk, pur = _has(ss["clicked"]), _has(ss["purchased"])
    ctxs = ss["context"].map(_ctx)
    dev = ctxs.map(lambda x: x.get("device"))
    page = ctxs.map(lambda x: x.get("page"))
    nr = len(ss)
    f["n_requests"], f["n_carousel_clicks"], f["n_carousel_buys"] = nr, int(clk.sum()), int(pur.sum())
    f["carousel_ctr"] = clk.sum() / nr if nr > 0 else 0.0
    f["carousel_buy_rate"] = pur.sum() / nr if nr > 0 else 0.0
    f["days_since_last_request"] = sage.min() if nr > 0 else 999.0
    f["n_requests_last_7d"] = int((sage <= 7).sum())
    
    for d in DEVICES:
        f[f"device_share_{d}"] = float((dev == d).mean()) if nr > 0 else 0.0
    f["page_product_share"] = float((page == "product").mean()) if nr > 0 else 0.0

    return {k: float(x) for k, x in f.items()}


# 3) la funcion publica
def user_features(user_id, ts):
    """user_id + ts de la peticion -> dict {nombre_feature: valor}."""
    ev, ss = fetch_user_history(user_id, ts)
    return build_user_features(ev, ss, ts)