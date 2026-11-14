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


def _is_match(product_id: str, target_val) -> int:
    """Safely checks if product_id matches a single target ID, sequence of IDs, or NaN."""
    if target_val is None or (isinstance(target_val, float) and np.isnan(target_val)):
        return 0
    if isinstance(target_val, (list, tuple, set, np.ndarray)):
        return int(product_id in target_val)
    return int(product_id == target_val)


class FeatureConstructor:
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
    ):
        self._initialized = False
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

        self._initialized = True

    def constructFeatures(
        self,
        data: dict,
        use_coview_features: bool = None,
        use_user_features: bool = None,
        use_session_features: bool = None,
        use_cross_features: bool = None,
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
            "is_active": int(bool(data.get("is_active"))),
            "price_eur": price_eur,
            "product_age_days": product_age_days,
            "description_len": len(description),
            "is_bestseller": int(product_id in self.best_sellers_set),
            "is_out_of_stock": int(product_id in self.out_of_stock_set),
            "is_new_user": int(bool(data.get("is_new_user"))),
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
        }

        # 5. Session-level & last-viewed item comparison features (if enabled)
        if session_on:
            session_length = len(recent_views)
            same_category_as_last_view = 0
            same_brand_as_last_view = 0
            price_diff_from_last_view = 0.0

            if session_length > 0 and self.catalog_lookup:
                last_event = recent_views[-1]
                last_viewed_id = (
                    last_event.get("product_id")
                    if isinstance(last_event, dict)
                    else last_event
                )
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

        # 7. Cross-features combining user history with current candidate item (if enabled)
        if cross_on and user_on:
            user_avg_price = user_data.get("avg_price_eur") or user_data.get("user_avg_price")
            if user_avg_price and user_avg_price > 0:
                X["price_ratio_to_user_avg"] = price_eur / float(user_avg_price)
                X["price_diff_from_user_avg"] = price_eur - float(user_avg_price)
            else:
                X["price_ratio_to_user_avg"] = 1.0
                X["price_diff_from_user_avg"] = 0.0

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

        user_cache = {}
        X_rows = []
        y_click = []
        y_buy = []
        sample_weights = []

        iterator = tqdm(
            zip(interactions_df.itertuples(), parsed_ts, hours, dows),
            total=len(interactions_df),
            desc="Generating features",
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
            is_new_user = int(ts_raw <= first_seen_ts.get(user_id, ts_raw))
            is_weekend = int(day_of_week in (5, 6)) if day_of_week != -1 else 0

            # 1. Fetch user features ONCE per (user_id, ts) instead of once per shown item
            user_data = {}
            user_avg_price = None
            if user_on:
                cache_key = (user_id, ts_raw)
                if cache_key not in user_cache:
                    user_cache[cache_key] = userFeatures.user_features(user_id, ts_raw) or {}
                user_data = user_cache[cache_key]
                if cross_on:
                    user_avg_price = user_data.get("avg_price_eur") or user_data.get("user_avg_price")

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
                last_event = recent_views[-1]
                last_viewed_id = last_event.get("product_id") if isinstance(last_event, dict) else last_event
                last_meta = self.catalog_lookup.get(last_viewed_id)

            clicked_val = getattr(row, "clicked", None)
            purchased_val = getattr(row, "purchased", None)

            # 3. Inner loop now only computes item-specific features
            for prod_id, prop in zip(shown_items, propensities):
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
                    "is_active": int(bool(prod_meta.get("is_active"))),
                    "price_eur": price_eur,
                    "product_age_days": product_age_days,
                    "description_len": len(description),
                    "is_bestseller": int(prod_id in self.best_sellers_set),
                    "is_out_of_stock": int(prod_id in self.out_of_stock_set),
                    "is_new_user": is_new_user,
                    "hour_of_day": hour_of_day,
                    "day_of_week": day_of_week,
                    "is_weekend": is_weekend,
                }

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

                if cross_on and user_on:
                    if user_avg_price and user_avg_price > 0:
                        X["price_ratio_to_user_avg"] = price_eur / float(user_avg_price)
                        X["price_diff_from_user_avg"] = price_eur - float(user_avg_price)
                    else:
                        X["price_ratio_to_user_avg"] = 1.0
                        X["price_diff_from_user_avg"] = 0.0

                X_rows.append(X)
                y_click.append(_is_match(prod_id, clicked_val))
                y_buy.append(_is_match(prod_id, purchased_val))

                prop_val = float(prop) if prop is not None and pd.notna(prop) else 1.0
                sample_weights.append(1.0 / max(prop_val, 0.01))

        X_df = pd.DataFrame(X_rows)
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


# Backwards-compatible module-level singleton & functions
_default_constructor = FeatureConstructor()


def constructFeatures(data: dict, **kwargs) -> dict:
    return _default_constructor.constructFeatures(data, **kwargs)


def generate_xy(*args, **kwargs):
    return _default_constructor.generate_xy(*args, **kwargs)