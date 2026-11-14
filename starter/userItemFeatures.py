"""
userItemFeatures.py - features usuario x producto a partir de los embeddings iALS.

Uso:
    import userItemFeatures as uif
    uif.load()                                   # busca ../../models y models; ya no carga nada al importar
    uif.ials_features("u_123", ["p_a", "p_b"])   # {pid: {ials_score, ials_rank, ials_score_centered, ials_user_known}}
    uif.user_item_features("u_123")              # top-10 (compatibilidad con la version anterior)

Idea (para la defensa): iALS da a cada usuario y a cada producto un "perfil" de 32 numeros.
Cuanto mas se parecen (producto escalar), mas encaja el producto con ese usuario.
"""
import pickle
from pathlib import Path

import numpy as np

_S = {}


def load(models_dir=None):
    """Carga factores y mapeos. Devuelve True si hubo que intercambiar usuarios/productos."""
    cands = [Path(models_dir)] if models_dir else [Path("../../models"), Path("models")]
    d = next((c for c in cands if (c / "ials_model.npz").exists()), None)
    if d is None:
        raise FileNotFoundError(f"No encuentro ials_model.npz en {[str(c) for c in cands]}")
    z = np.load(d / "ials_model.npz")
    U, I = np.asarray(z["user_factors"]), np.asarray(z["item_factors"])
    with open(d / "mappings.pkl", "rb") as f:
        m = pickle.load(f)
    u2i, p2i = m["user_to_idx"], m["product_to_idx"]
    idx2p = m.get("idx_to_product") or {i: p for p, i in p2i.items()}

    # En implicit >= 0.5, fit() espera usuarios x productos. Si se entreno con la traspuesta,
    # user_factors e item_factors salen intercambiados: lo detectamos por la forma y lo corregimos.
    swapped = False
    if U.shape[0] == len(p2i) and I.shape[0] == len(u2i) and len(p2i) != len(u2i):
        U, I, swapped = I, U, True
    assert U.shape[0] == len(u2i) and I.shape[0] == len(p2i), (
        f"Formas incoherentes: user_factors{U.shape}, item_factors{I.shape}, "
        f"{len(u2i)} usuarios, {len(p2i)} productos")
    _S.update(U=U, I=I, u2i=u2i, p2i=p2i, idx2p=idx2p)
    if swapped:
        print("AVISO: user_factors/item_factors estaban intercambiados; corregido al cargar. "
              "Reentrena con model.fit(user_item_matrix) (sin .T) para dejarlo bien en disco.")
    return swapped


def ials_features(user_id, product_ids):
    """{product_id: {ials_score, ials_rank, ials_score_centered, ials_user_known}} para una lista de candidatos."""
    assert _S, "Llama primero a userItemFeatures.load()"
    pids = [str(p) for p in product_ids]
    u = _S["u2i"].get(str(user_id))
    scores = np.full(len(pids), np.nan)
    if u is not None:
        uvec = _S["U"][u]
        for k, p in enumerate(pids):
            j = _S["p2i"].get(p)
            if j is not None:
                scores[k] = float(uvec @ _S["I"][j])
    ok = ~np.isnan(scores)
    ranks = np.full(len(pids), np.nan)
    if ok.any():
        ranks[ok] = (-scores[ok]).argsort().argsort() + 1
    mean = np.nanmean(scores) if ok.any() else np.nan
    return {p: {"ials_score": float(scores[k]), "ials_rank": float(ranks[k]),
                "ials_score_centered": float(scores[k] - mean) if ok.any() else np.nan,
                "ials_user_known": int(u is not None)}
            for k, p in enumerate(pids)}


def user_item_features(sample_user_id, N=10):
    """Top-N productos para un usuario como lista [(product_id, score)]. Cold start -> lista vacia."""
    assert _S, "Llama primero a userItemFeatures.load()"
    u = _S["u2i"].get(str(sample_user_id))
    if u is None:
        return []
    sc = _S["I"] @ _S["U"][u]
    top = np.argsort(-sc)[:N]
    return [(_S["idx2p"][int(i)], float(sc[i])) for i in top]
