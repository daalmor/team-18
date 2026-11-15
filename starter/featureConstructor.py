import warnings

import numpy as np
import pandas as pd
from coviewFeatures import extract_coview_features_from_events
import userFeatures

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def merge(dict1: dict, dict2: dict) -> dict:
    """Merges two dictionaries with dict2 taking precedence."""
    return {**dict1, **dict2}


def _extract_id_set(data) -> set:
    """Safely extracts product IDs from a DataFrame (including RangeIndex), Series, or iterable."""
    if data is None:
        return set()
    if isinstance(data, pd.DataFrame):
        if data.empty:
            return set()
        if "product_id" in data.columns:
            return set(data["product_id"].dropna())
        if data.index.name == "product_id":
            return set(data.index.dropna())
        return set(data.iloc[:, 0].dropna())
    if isinstance(data, pd.Series):
        return set(data.dropna())
    return set(data)


def _last_viewed_id(recent_views):
    """Id del producto visto mas recientemente (por ts, sin asumir el orden de la lista)."""
    if not recent_views:
        return None
    last = recent_views[-1]
    if all(isinstance(v, dict) and v.get("ts") is not None for v in recent_views):
        last = max(recent_views, key=lambda v: str(v["ts"]))
    return last.get("product_id") if isinstance(last, dict) else last


def _is_match(product_id: str, target_val) -> int:
    """Safely checks if product_id matches a single target ID, sequence of IDs, or NaN."""
    if target_val is None or (isinstance(target_val, float) and np.isnan(target_val)):
        return 0
    if isinstance(target_val, (list, tuple, set, np.ndarray)):
        return int(product_id in target_val)
    return int(product_id == target_val)


def _cross_features(user_data: dict, category, price_eur: float) -> dict:
    """Features que cruzan el historial del usuario con UN candidato concreto."""
    nan = float("nan")
    lp_user = user_data.get("price_log_mean", nan)
    mean_user = user_data.get("price_mean_eur", nan)
    ok_lp = lp_user is not None and not pd.isna(lp_user)
    ok_mean = mean_user is not None and not pd.isna(mean_user) and mean_user > 0
    share = user_data.get(f"cat_share_{category}", nan)
    return {
        "price_log_gap": float(np.log1p(price_eur) - lp_user) if ok_lp else nan,        # >0: mas caro de lo habitual
        "price_ratio_to_user_avg": float(price_eur / mean_user) if ok_mean else nan,
        "price_diff_from_user_avg": float(price_eur - mean_user) if ok_mean else nan,
        "user_cat_share_of_candidate": float(share) if share is not None else nan,     # afinidad con su categoria
    }


class FeatureConstructor:
    INCLUDE_STATE_FLAGS = False   # is_active / is_bestseller / is_out_of_stock: constantes o con fuga en entrenamiento
    CATEGORICAL_COLS = ["brand", "category", "subcategory", "device", "region", "page"]

    def __init__(
        self,
        query_function=None,
        catalog: pd.DataFrame = None,
        coview_dict: dict = None,
        best_sellers=None,
        out_of_stock=None,
        use_coview_features: bool = True,
        use_user_features: bool = True,
        use_session_features: bool = True,
        use_cross_features: bool = True,
        use_ials_features: bool = False,   # score iALS usuario x producto (requiere models/ials_model.npz)
        models_dir: str = None,
        popularity: dict = None,           # {product_id: {pop_* features}} (ver popularity.load_popularity)
    ):
        self._initialized = False
        self.pop_lookup = popularity or {}
        self.feature_columns = None     # columnas del modelo (ver set_schema)
        self.category_dtypes = {}       # categorias de entrenamiento (ver set_schema)
        self.column_dtypes = {}         # tipos numericos de entrenamiento (ver set_schema)
        self.query_function = query_function
        self.catalog = catalog
        self.coview_dict = coview_dict or {}
        self.catalog_lookup = {}
        self.best_sellers_set = set()
        self.out_of_stock_set = set()

        # Feature group toggles
        self.use_coview_features = use_coview_features
        self.use_user_features = use_user_features
        self.use_session_features = use_session_features
        self.use_cross_features = use_cross_features
        self.use_ials_features = use_ials_features
        self.models_dir = models_dir

        if catalog is not None or query_function is not None:
            self.init_dependencies(
                query_function, catalog, coview_dict, best_sellers, out_of_stock
            )

    def init_dependencies(
        self,
        query_function,
        catalog: pd.DataFrame,
        coview_dict: dict = None,
        best_sellers=None,
        out_of_stock=None,
    ):
        """Initializes heavy lookups once instead of repeating on every candidate row."""
        self.query_function = query_function
        self.catalog = catalog
        if coview_dict is not None:
            self.coview_dict = coview_dict

        if self.use_user_features and query_function is not None and catalog is not None:
            userFeatures.init(query_function, catalog)

        if catalog is not None and not catalog.empty:
            cat_df = (
                catalog.set_index("product_id")
                if "product_id" in catalog.columns
                else catalog
            )
            self.catalog_lookup = cat_df.to_dict("index")

        self.best_sellers_set = _extract_id_set(best_sellers)
        self.out_of_stock_set = _extract_id_set(out_of_stock)

        if self.use_ials_features:
            import userItemFeatures
            userItemFeatures.load(self.models_dir)

        self._initialized = True

    def _pop(self, product_id) -> dict:
        """Popularidad del producto (o los valores por defecto si no hay estadisticas)."""
        return dict(self.pop_lookup.get(product_id) or self.pop_lookup.get("__default__") or {})

    def constructFeatures(
        self,
        data: dict,
        use_coview_features: bool = None,
        use_user_features: bool = None,
        use_session_features: bool = None,
        use_cross_features: bool = None,
        use_ials_features: bool = None,
    ) -> dict:
        """
        Constructs GBM-ready features from a single candidate payload dictionary.
        Optional boolean flags override the instance-level feature toggles for this call.
        """
        coview_on = self.use_coview_features if use_coview_features is None else use_coview_features
        user_on = self.use_user_features if use_user_features is None else use_user_features
        session_on = self.use_session_features if use_session_features is None else use_session_features
        cross_on = self.use_cross_features if use_cross_features is None else use_cross_features

        if not self._initialized:
            self.init_dependencies(
                data.get("query_function"),
                data.get("catalog"),
                data.get("coview_dict"),
                data.get("best_sellers"),
                data.get("out_of_stock"),
            )

        product_id = data.get("product_id")
        user_id = data.get("user_id")
        ts_raw = data.get("ts")
        created_at_raw = data.get("created_at")

        recent_views = data.get("recent_views")
        if isinstance(recent_views, np.ndarray):
            recent_views = recent_views.tolist()
        elif not isinstance(recent_views, list):
            recent_views = []

        coview_dict = data.get("coview_dict") or self.coview_dict

        # 1. Fetch external user features (if enabled)
        user_data = {}
        if user_on:
            user_data = userFeatures.user_features(user_id, ts_raw) or {}

        # 2. Parse timestamps into numeric GBM features
        req_ts = pd.to_datetime(ts_raw, errors="coerce")
        created_ts = pd.to_datetime(created_at_raw, errors="coerce")

        product_age_days = (
            max((req_ts - created_ts).total_seconds() / 86400.0, 0.0)
            if pd.notna(req_ts) and pd.notna(created_ts)
            else -1.0
        )
        hour_of_day = int(req_ts.hour) if pd.notna(req_ts) else -1
        day_of_week = int(req_ts.dayofweek) if pd.notna(req_ts) else -1
        is_weekend = int(day_of_week in (5, 6)) if day_of_week != -1 else 0

        # 3. Extract context safely
        context = data.get("context")
        get_ctx = (
            lambda attr: getattr(context, attr, None)
            if hasattr(context, attr)
            else (context.get(attr) if isinstance(context, dict) else None)
        )

        # 4. Base product, catalog status, temporal, and context features
        description = data.get("description") or ""
        price_eur = float(data.get("price_eur") or 0.0)

        X = {
            "brand": data.get("brand"),
            "category": data.get("category"),
            "subcategory": data.get("subcategory"),
            "device": get_ctx("device"),
            "region": get_ctx("region"),
            "page": get_ctx("page"),
            "price_eur": price_eur,
            "product_age_days": product_age_days,
            "description_len": len(description),
            "is_new_user": int(bool(data.get("is_new_user"))),
            "position": int(data.get("position", 0)),   # al servir: 0 = "como si fuera el primer hueco"
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
        }

        if self.INCLUDE_STATE_FLAGS:
            X["is_active"] = int(bool(data.get("is_active")))
            X["is_bestseller"] = int(product_id in self.best_sellers_set)
            X["is_out_of_stock"] = int(product_id in self.out_of_stock_set)

        # 5. Session-level & last-viewed item comparison features (if enabled)
        if session_on:
            session_length = len(recent_views)
            same_category_as_last_view = 0
            same_brand_as_last_view = 0
            price_diff_from_last_view = 0.0

            if session_length > 0 and self.catalog_lookup:
                last_viewed_id = _last_viewed_id(recent_views)
                last_meta = self.catalog_lookup.get(last_viewed_id)
                if last_meta:
                    same_category_as_last_view = int(
                        data.get("category") == last_meta.get("category")
                    )
                    same_brand_as_last_view = int(
                        data.get("brand") == last_meta.get("brand")
                    )
                    if (
                        data.get("price_eur") is not None
                        and last_meta.get("price_eur") is not None
                    ):
                        price_diff_from_last_view = float(data["price_eur"]) - float(
                            last_meta["price_eur"]
                        )

            X["session_length"] = session_length
            X["same_category_as_last_view"] = same_category_as_last_view
            X["same_brand_as_last_view"] = same_brand_as_last_view
            X["price_diff_from_last_view"] = price_diff_from_last_view

        # 6. Merge enabled feature dictionaries
        if coview_on:
            X = merge(
                X,
                extract_coview_features_from_events(
                    product_id,
                    recent_views,
                    coview_dict,
                ),
            )
        if user_on:
            X = merge(X, user_data)
        if self.pop_lookup:
            X.update(self._pop(product_id))

        ials_on = self.use_ials_features if use_ials_features is None else use_ials_features
        if ials_on:
            import userItemFeatures
            r_ = userItemFeatures.ials_features(user_id, [product_id])[str(product_id)]
            X["ials_score"], X["ials_user_known"] = r_["ials_score"], r_["ials_user_known"]

        # 7. Cross-features usuario x candidato (si estan activas)
        if cross_on and user_on:
            X.update(_cross_features(user_data, data.get("category"), price_eur))

        return X

    def generate_xy(
        self,
        interactions_df: pd.DataFrame,
        coview_dict: dict = None,
        history_sessions: pd.DataFrame = None,
        query_function=None,
        catalog: pd.DataFrame = None,
        best_sellers=None,
        out_of_stock=None,
        cast_categoricals: bool = True,
        use_coview_features: bool = None,
        use_user_features: bool = None,
        use_session_features: bool = None,
        use_cross_features: bool = None,
        show_progress: bool = True,
        use_ials_features: bool = None,
        prefetch_history: bool = True,
    ):
        """
        Generates (X, y_click, y_buy, sample_weights) training pairs from session logs.
        Optimized to compute user and session features once per session instead of per item.
        """
        if not self._initialized:
            self.init_dependencies(
                query_function or self.query_function,
                catalog if catalog is not None else self.catalog,
                coview_dict or self.coview_dict,
                best_sellers,
                out_of_stock,
            )

        coview_on = self.use_coview_features if use_coview_features is None else use_coview_features
        user_on = self.use_user_features if use_user_features is None else use_user_features
        session_on = self.use_session_features if use_session_features is None else use_session_features
        cross_on = self.use_cross_features if use_cross_features is None else use_cross_features

        ials_on = self.use_ials_features if use_ials_features is None else use_ials_features
        if ials_on:
            import userItemFeatures

        active_coview_dict = coview_dict if coview_dict is not None else self.coview_dict

        ref_sessions = history_sessions if history_sessions is not None else interactions_df
        first_seen_ts = ref_sessions.groupby("user_id")["ts"].min().to_dict()

        # Pre-parse timestamps once in vectorized Pandas instead of per-row
        parsed_ts = pd.to_datetime(interactions_df["ts"], errors="coerce")
        hours = parsed_ts.dt.hour.fillna(-1).astype(int).tolist()
        dows = parsed_ts.dt.dayofweek.fillna(-1).astype(int).tolist()

        # Pre-parse catalog created_at once
        created_at_map = {}
        if self.catalog is not None and "created_at" in self.catalog.columns:
            cat_idx = self.catalog.set_index("product_id") if "product_id" in self.catalog.columns else self.catalog
            created_at_map = pd.to_datetime(cat_idx["created_at"], errors="coerce").to_dict()

        # Descarga el historial de TODOS los usuarios de golpe (2 consultas por lote, no 2 por peticion)
        if user_on and prefetch_history:
            userFeatures.prefetch(interactions_df["user_id"].unique(), parsed_ts.max(), show_progress=show_progress)

        user_cache = {}
        meta_rows = []
        X_rows = []
        y_click = []
        y_buy = []
        sample_weights = []

        iterator = tqdm(
            zip(interactions_df.itertuples(), parsed_ts, hours, dows),
            total=len(interactions_df),
            desc="Generando features",
            disable=not show_progress,
        )

        for row, req_ts, hour_of_day, day_of_week in iterator:
            shown_items = getattr(row, "shown", None)
            if not isinstance(shown_items, (list, tuple, np.ndarray)) or len(shown_items) == 0:
                continue

            propensities = getattr(row, "propensity", None)
            if not isinstance(propensities, (list, tuple, np.ndarray)) or len(propensities) != len(shown_items):
                propensities = [1.0] * len(shown_items)

            user_id = row.user_id
            ts_raw = row.ts
            is_new_user = 0   # se sobrescribe abajo con user_data["is_new_user"] (sin eventos previos)
            is_weekend = int(day_of_week in (5, 6)) if day_of_week != -1 else 0

            # 1. Fetch user features ONCE per (user_id, ts) instead of once per shown item
            user_data = {}
            if user_on:
                cache_key = (user_id, ts_raw)
                if cache_key not in user_cache:
                    user_cache[cache_key] = userFeatures.user_features(user_id, ts_raw) or {}
                user_data = user_cache[cache_key]
                is_new_user = int(user_data.get("is_new_user", is_new_user))

            # 2. Extract context & recent_views ONCE per session
            context = getattr(row, "context", None)
            if isinstance(context, dict):
                device, region, page = context.get("device"), context.get("region"), context.get("page")
            elif context is not None:
                device = getattr(context, "device", None)
                region = getattr(context, "region", None)
                page = getattr(context, "page", None)
            else:
                device = region = page = None

            recent_views = getattr(row, "recent_views", None)
            if isinstance(recent_views, np.ndarray):
                recent_views = recent_views.tolist()
            elif not isinstance(recent_views, list):
                recent_views = []

            session_length = len(recent_views)
            last_meta = None
            if session_on and session_length > 0 and self.catalog_lookup:
                last_viewed_id = _last_viewed_id(recent_views)
                last_meta = self.catalog_lookup.get(last_viewed_id)

            ials_by_prod = userItemFeatures.ials_features(user_id, shown_items) if ials_on else {}
            mult = getattr(row, "sample_mult", 1.0)    # >1 si esta peticion representa a varias (muestreo)
            mult = 1.0 if mult is None or pd.isna(mult) else float(mult)

            clicked_val = getattr(row, "clicked", None)
            purchased_val = getattr(row, "purchased", None)

            # 3. Inner loop now only computes item-specific features
            for pos_i, (prod_id, prop) in enumerate(zip(shown_items, propensities)):
                prod_meta = self.catalog_lookup.get(prod_id)
                if prod_meta is None:
                    continue

                created_ts = created_at_map.get(prod_id)
                product_age_days = (
                    max((req_ts - created_ts).total_seconds() / 86400.0, 0.0)
                    if pd.notna(req_ts) and pd.notna(created_ts)
                    else -1.0
                )

                description = prod_meta.get("description") or ""
                price_eur = float(prod_meta.get("price_eur") or 0.0)

                X = {
                    "brand": prod_meta.get("brand"),
                    "category": prod_meta.get("category"),
                    "subcategory": prod_meta.get("subcategory"),
                    "device": device,
                    "region": region,
                    "page": page,
                    "price_eur": price_eur,
                    "product_age_days": product_age_days,
                    "description_len": len(description),
                    "is_new_user": is_new_user,
                    "position": pos_i,
                    "hour_of_day": hour_of_day,
                    "day_of_week": day_of_week,
                    "is_weekend": is_weekend,
                }

                if self.INCLUDE_STATE_FLAGS:
                    X["is_active"] = int(bool(prod_meta.get("is_active")))
                    X["is_bestseller"] = int(prod_id in self.best_sellers_set)
                    X["is_out_of_stock"] = int(prod_id in self.out_of_stock_set)

                if session_on:
                    if last_meta:
                        same_cat = int(prod_meta.get("category") == last_meta.get("category"))
                        same_brand = int(prod_meta.get("brand") == last_meta.get("brand"))
                        price_diff = (
                            price_eur - float(last_meta["price_eur"])
                            if prod_meta.get("price_eur") is not None and last_meta.get("price_eur") is not None
                            else 0.0
                        )
                    else:
                        same_cat, same_brand, price_diff = 0, 0, 0.0

                    X["session_length"] = session_length
                    X["same_category_as_last_view"] = same_cat
                    X["same_brand_as_last_view"] = same_brand
                    X["price_diff_from_last_view"] = price_diff

                if coview_on:
                    X.update(extract_coview_features_from_events(prod_id, recent_views, active_coview_dict))
                if user_on:
                    X.update(user_data)
                if self.pop_lookup:
                    X.update(self._pop(prod_id))

                if cross_on and user_on:
                    X.update(_cross_features(user_data, prod_meta.get("category"), price_eur))

                if ials_on:
                    r_ = ials_by_prod.get(str(prod_id), {})
                    X["ials_score"] = r_.get("ials_score", float("nan"))
                    X["ials_user_known"] = r_.get("ials_user_known", 0)

                X_rows.append(X)
                meta_rows.append((getattr(row, "request_id", None), ts_raw, mult))
                y_click.append(_is_match(prod_id, clicked_val))
                y_buy.append(_is_match(prod_id, purchased_val))

                prop_val = float(prop) if prop is not None and pd.notna(prop) else 1.0
                sample_weights.append(mult / max(prop_val, 0.05))

        X_df = pd.DataFrame(X_rows)
        # request_id / ts / sample_mult de cada fila de X (para validar por peticion)
        self.last_meta = pd.DataFrame(meta_rows, columns=["request_id", "ts", "sample_mult"])
        if cast_categoricals and not X_df.empty:
            for col in self.CATEGORICAL_COLS:
                if col in X_df.columns:
                    X_df[col] = X_df[col].astype("category")

        return (
            X_df,
            np.array(y_click, dtype=np.int32),
            np.array(y_buy, dtype=np.int32),
            np.array(sample_weights, dtype=np.float32),
        )

    # ------------------------------------------------------------------ PREDICCION (una fila)
    def set_schema(self, X: pd.DataFrame, model_columns=None):
        """Guarda las columnas y las categorias del entrenamiento para que la fila de prediccion encaje
        EXACTAMENTE con lo que vio el modelo. Llamar una vez tras generate_xy:  fc.set_schema(X)"""
        self.feature_columns = list(model_columns) if model_columns is not None else list(X.columns)
        self.column_dtypes = {c: X[c].dtype for c in X.columns if str(X[c].dtype) != "category"}
        self.category_dtypes = {
            c: X[c].dtype for c in self.CATEGORICAL_COLS if c in X.columns and str(X[c].dtype) == "category"
        }

    def build_prediction_row(
        self,
        brand=None,
        category=None,
        subcategory=None,
        created_at=None,
        description=None,
        is_new_user=False,
        device=None,
        region=None,
        page=None,
        *,
        price_eur=None,
        product_id=None,
        user_id=None,
        ts=None,
        recent_views=None,
        position: int = 0,
    ) -> pd.DataFrame:
        """
        Devuelve un DataFrame de 1 fila (X) listo para model.predict_proba(...) con UN producto candidato.

        - Si is_new_user=True: no se consulta nada; las features de usuario toman sus valores por defecto
          (los mismos que produce userFeatures para alguien sin historial: contadores a 0, tiempos a 999, precios NaN).
        - Si is_new_user=False: hace falta user_id (y ts; si no se da, se usa "ahora") para calcular su historial.
        - product_id es opcional: si se da, lo que falte (brand, category, price_eur...) se toma del catalogo, y se
          usa para las features de coview / iALS. recent_views = lista de {"product_id", "ts"}.
        - position: 0 al servir ("como si fuera el primer hueco").
        """
        import userFeatures

        meta = self.catalog_lookup.get(product_id, {}) if product_id is not None else {}
        pick = lambda v, k: meta.get(k) if v is None else v
        brand, category, subcategory = pick(brand, "brand"), pick(category, "category"), pick(subcategory, "subcategory")
        created_at, description = pick(created_at, "created_at"), pick(description, "description")
        price_eur = pick(price_eur, "price_eur")
        price_eur = float("nan") if price_eur is None else float(price_eur)   # sin precio -> NaN, no 0 euros

        ts = pd.Timestamp.now(tz="UTC").tz_localize(None) if ts is None else ts
        data = {
            "product_id": product_id, "user_id": user_id, "ts": ts, "created_at": created_at,
            "brand": brand, "category": category, "subcategory": subcategory, "description": description,
            "price_eur": price_eur, "is_new_user": bool(is_new_user), "position": position,
            "context": {"device": device, "region": region, "page": page},
            "recent_views": recent_views if recent_views is not None else [],
        }

        if is_new_user:
            # sin historial: se calculan todas las features salvo las de usuario, y estas se rellenan con sus defaults
            row = self.constructFeatures(data, use_user_features=False, use_cross_features=False)
            if self.use_user_features:
                if getattr(userFeatures, "_cat", None) is None:
                    userFeatures.init(self.query_function or (lambda sql: pd.DataFrame()), self.catalog)
                no_hist = pd.DataFrame({c: [] for c in ["session_id", "event_type", "product_id", "ts"]})
                no_sess = pd.DataFrame({c: [] for c in ["request_id", "ts", "context", "clicked", "purchased"]})
                defaults = userFeatures.build_user_features(no_hist, no_sess, ts)
                row.update(defaults)
                row["is_new_user"] = 1
                if self.use_cross_features:
                    row.update(_cross_features(defaults, category, price_eur))
        else:
            if user_id is None:
                raise ValueError("is_new_user=False: hace falta user_id para calcular su historial")
            row = self.constructFeatures(data)

        X = pd.DataFrame([row])
        if self.feature_columns is not None:
            missing = [c for c in self.feature_columns if c not in X.columns]
            if missing:
                warnings.warn(f"Columnas del modelo que no se pudieron calcular (van a NaN): {missing}")
            X = X.reindex(columns=self.feature_columns)
            for c, dt in self.column_dtypes.items():
                if c in X.columns:
                    try:
                        X[c] = X[c].astype(dt)
                    except (ValueError, TypeError):
                        pass
        for c, dt in self.category_dtypes.items():
            if c in X.columns:
                if X[c].notna().any() and not X[c].isin(dt.categories).all():
                    warnings.warn(f"'{c}' = {X[c].iloc[0]!r} no existia en el entrenamiento (se trata como NaN)")
                X[c] = pd.Categorical(X[c], categories=dt.categories)
        return X


# Backwards-compatible module-level singleton & functions
_default_constructor = FeatureConstructor()


def constructFeatures(data: dict, **kwargs) -> dict:
    return _default_constructor.constructFeatures(data, **kwargs)


def generate_xy(*args, **kwargs):
    return _default_constructor.generate_xy(*args, **kwargs)