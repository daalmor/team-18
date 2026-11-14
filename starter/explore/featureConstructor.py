import numpy as np
import pandas as pd
from coviewFeatures import extract_coview_features_from_events
import userFeatures
import userItemFeatures


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
        use_user_item_features: bool = True,
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
        self.use_user_item_features = use_user_item_features
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
        use_user_item_features: bool = None,
        use_session_features: bool = None,
        use_cross_features: bool = None,
    ) -> dict:
        """
        Constructs GBM-ready features from a single candidate payload dictionary.
        Optional boolean flags override the instance-level feature toggles for this call.
        """
        coview_on = self.use_coview_features if use_coview_features is None else use_coview_features
        user_on = self.use_user_features if use_user_features is None else use_user_features
        user_item_on = self.use_user_item_features if use_user_item_features is None else use_user_item_features
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

        # 1. Fetch external user & user-item features (if enabled)
        user_data = {}
        if user_on:
            user_data = userFeatures.user_features(user_id, ts_raw) or {}

        user_item_data = {}
        if user_item_on:
            try:
                user_item_data = userItemFeatures.user_item_features(user_id, product_id) or {}
            except TypeError:
                raw_ui = userItemFeatures.user_item_features(user_id) or {}
                user_item_data = (
                    raw_ui.get(product_id, {})
                    if isinstance(raw_ui, dict) and product_id in raw_ui
                    else raw_ui
                )

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
        if user_item_on:
            X = merge(X, user_item_data)

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
        use_user_item_features: bool = None,
        use_session_features: bool = None,
        use_cross_features: bool = None,
    ):
        """
        Generates (X, y_click, y_buy, sample_weights) training pairs from session logs.
        """
        if not self._initialized:
            self.init_dependencies(
                query_function or self.query_function,
                catalog if catalog is not None else self.catalog,
                coview_dict or self.coview_dict,
                best_sellers,
                out_of_stock,
            )

        active_coview_dict = coview_dict if coview_dict is not None else self.coview_dict

        ref_sessions = history_sessions if history_sessions is not None else interactions_df
        first_seen_ts = ref_sessions.groupby("user_id")["ts"].min().to_dict()

        X_rows = []
        y_click = []
        y_buy = []
        sample_weights = []

        for row in interactions_df.itertuples():
            shown_items = getattr(row, "shown", None)
            if not isinstance(shown_items, (list, tuple, np.ndarray)) or len(shown_items) == 0:
                continue

            propensities = getattr(row, "propensity", None)
            if (
                not isinstance(propensities, (list, tuple, np.ndarray))
                or len(propensities) != len(shown_items)
            ):
                propensities = [1.0] * len(shown_items)

            is_new_user = row.ts <= first_seen_ts.get(row.user_id, row.ts)
            recent_views = getattr(row, "recent_views", None)
            if isinstance(recent_views, np.ndarray):
                recent_views = recent_views.tolist()
            elif not isinstance(recent_views, list):
                recent_views = []

            for prod_id, prop in zip(shown_items, propensities):
                prod_meta = self.catalog_lookup.get(prod_id)
                if prod_meta is None:
                    continue

                payload = {
                    "product_id": prod_id,
                    "brand": prod_meta.get("brand"),
                    "category": prod_meta.get("category"),
                    "created_at": prod_meta.get("created_at"),
                    "description": prod_meta.get("description"),
                    "is_active": prod_meta.get("is_active"),
                    "price_eur": prod_meta.get("price_eur"),
                    "subcategory": prod_meta.get("subcategory"),
                    "ts": row.ts,
                    "user_id": row.user_id,
                    "is_new_user": is_new_user,
                    "context": getattr(row, "context", None),
                    "recent_views": recent_views,
                    "coview_dict": active_coview_dict,
                    "query_function": self.query_function,
                    "catalog": self.catalog,
                }

                X_rows.append(
                    self.constructFeatures(
                        payload,
                        use_coview_features=use_coview_features,
                        use_user_features=use_user_features,
                        use_user_item_features=use_user_item_features,
                        use_session_features=use_session_features,
                        use_cross_features=use_cross_features,
                    )
                )
                y_click.append(_is_match(prod_id, getattr(row, "clicked", None)))
                y_buy.append(_is_match(prod_id, getattr(row, "purchased", None)))

                prop_val = (
                    float(prop)
                    if prop is not None and pd.notna(prop)
                    else 1.0
                )
                sample_weights.append(1.0 / max(prop_val, 0.01))

        X = pd.DataFrame(X_rows)
        if cast_categoricals and not X.empty:
            for col in self.CATEGORICAL_COLS:
                if col in X.columns:
                    X[col] = X[col].astype("category")

        return (
            X,
            np.array(y_click, dtype=np.int32),
            np.array(y_buy, dtype=np.int32),
            np.array(sample_weights, dtype=np.float32),
        )


# Backwards-compatible module-level singleton & functions
_default_constructor = FeatureConstructor()


def constructFeatures(data: dict, **kwargs) -> dict:
    return _default_constructor.constructFeatures(data, **kwargs)


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
        max_examples: int = None,
        random_state: int = None,
        use_coview_features: bool = None,
        use_user_features: bool = None,
        use_user_item_features: bool = None,
        use_session_features: bool = None,
        use_cross_features: bool = None,
    ):
        """
        Generates (X, y_click, y_buy, sample_weights) training pairs from session logs.
        
        Args:
            max_examples: Max number of (X, Y) candidate rows to generate (None = all).
            random_state: If set, shuffles sessions before sampling so examples span all timestamps.
        """
        if max_examples is not None and max_examples <= 0:
            return (
                pd.DataFrame(),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float32),
            )

        if not self._initialized:
            self.init_dependencies(
                query_function or self.query_function,
                catalog if catalog is not None else self.catalog,
                coview_dict or self.coview_dict,
                best_sellers,
                out_of_stock,
            )

        active_coview_dict = coview_dict if coview_dict is not None else self.coview_dict

        # Compute first interaction timestamp per user BEFORE any shuffling/slicing
        ref_sessions = history_sessions if history_sessions is not None else interactions_df
        first_seen_ts = ref_sessions.groupby("user_id")["ts"].min().to_dict()

        df_to_iter = (
            interactions_df.sample(frac=1.0, random_state=random_state)
            if random_state is not None
            else interactions_df
        )

        X_rows = []
        y_click = []
        y_buy = []
        sample_weights = []

        for row in df_to_iter.itertuples():
            if max_examples is not None and len(X_rows) >= max_examples:
                break

            shown_items = getattr(row, "shown", None)
            if not isinstance(shown_items, (list, tuple, np.ndarray)) or len(shown_items) == 0:
                continue

            propensities = getattr(row, "propensity", None)
            if (
                not isinstance(propensities, (list, tuple, np.ndarray))
                or len(propensities) != len(shown_items)
            ):
                propensities = [1.0] * len(shown_items)

            is_new_user = row.ts <= first_seen_ts.get(row.user_id, row.ts)
            recent_views = getattr(row, "recent_views", None)
            if isinstance(recent_views, np.ndarray):
                recent_views = recent_views.tolist()
            elif not isinstance(recent_views, list):
                recent_views = []

            for prod_id, prop in zip(shown_items, propensities):
                if max_examples is not None and len(X_rows) >= max_examples:
                    break

                prod_meta = self.catalog_lookup.get(prod_id)
                if prod_meta is None:
                    continue

                payload = {
                    "product_id": prod_id,
                    "brand": prod_meta.get("brand"),
                    "category": prod_meta.get("category"),
                    "created_at": prod_meta.get("created_at"),
                    "description": prod_meta.get("description"),
                    "is_active": prod_meta.get("is_active"),
                    "price_eur": prod_meta.get("price_eur"),
                    "subcategory": prod_meta.get("subcategory"),
                    "ts": row.ts,
                    "user_id": row.user_id,
                    "is_new_user": is_new_user,
                    "context": getattr(row, "context", None),
                    "recent_views": recent_views,
                    "coview_dict": active_coview_dict,
                    "query_function": self.query_function,
                    "catalog": self.catalog,
                }

                X_rows.append(
                    self.constructFeatures(
                        payload,
                        use_coview_features=use_coview_features,
                        use_user_features=use_user_features,
                        use_user_item_features=use_user_item_features,
                        use_session_features=use_session_features,
                        use_cross_features=use_cross_features,
                    )
                )
                y_click.append(_is_match(prod_id, getattr(row, "clicked", None)))
                y_buy.append(_is_match(prod_id, getattr(row, "purchased", None)))

                prop_val = (
                    float(prop)
                    if prop is not None and pd.notna(prop)
                    else 1.0
                )
                sample_weights.append(1.0 / max(prop_val, 0.01))

        X = pd.DataFrame(X_rows)
        if cast_categoricals and not X.empty:
            for col in self.CATEGORICAL_COLS:
                if col in X.columns:
                    X[col] = X[col].astype("category")

        return (
            X,
            np.array(y_click, dtype=np.int32),
            np.array(y_buy, dtype=np.int32),
            np.array(sample_weights, dtype=np.float32),
        )