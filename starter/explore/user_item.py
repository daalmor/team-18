import os
from pathlib import Path
import pandas as pd
import requests
import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares

# Environment setup
env_path = Path("../../.env") if Path("../../.env").exists() else Path(".env")
for line in env_path.read_text().splitlines() if env_path.exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
API_KEY = os.environ.get("RAWTREE_API_KEY")
TEAM = os.environ.get("TEAM_ID")

if not API_KEY:
    raise ValueError("RAWTREE_API_KEY is missing! Check your .env file location.")

http = requests.Session()
http.headers["Authorization"] = f"Bearer {API_KEY}"


def query(sql: str) -> pd.DataFrame:
    """Helper function to execute SQL queries against the RawTree API and return a DataFrame."""
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=600)
    if r.status_code != 200:
        print("--- CLICKHOUSE ERROR ---")
        print(r.text)
        print("------------------------")
    r.raise_for_status()
    data = r.json().get("data", [])
    return pd.DataFrame(data)


print("Fetching dataset cutoff date...")
last_day_df = query(
    "SELECT toDate(max(ts::DateTime64(3))) AS d FROM shop.history_sessions"
)
last_day = last_day_df.d[0]
CUT = str((pd.Timestamp(last_day) - pd.Timedelta(days=7)).date())
print(f"Train cutoff date (< CUT): {CUT}")

print("Fetching training interactions from RawTree...")

import pandas as pd

all_chunks = []
num_batches = 5  # Split your users into 5 parallel batches

for i in range(num_batches):
    print(f"Fetching user batch {i + 1} of {num_batches}...")

    chunk_df = query(f"""
        SELECT 
            user_id::String AS user_id, 
            product_id::String AS product_id,
            sum(
                CASE 
                    WHEN event_type = 'view' THEN 1
                    WHEN event_type = 'cart' THEN 3
                    WHEN event_type = 'purchase' THEN 5
                    ELSE 1
                END
            ) AS weight
        FROM shop.history_events
        WHERE ts < '{CUT}' 
          AND user_id != '' AND product_id != ''
          AND cityHash64(user_id) % {num_batches} = {i}
        GROUP BY user_id, product_id
    """)

    if not chunk_df.empty:
        all_chunks.append(chunk_df)

interactions_df = pd.concat(all_chunks, ignore_index=True)

print(f"Successfully loaded all interactions! Total pairs: {len(interactions_df)}")

# Aggregate weights by user and product
user_product_matrix_df = (
    interactions_df.groupby(["user_id", "product_id"])["weight"].sum().reset_index()
)

print(
    f"Loaded {len(user_product_matrix_df)} interactions across {user_product_matrix_df['user_id'].nunique()} users."
)


import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares

# 1. Create integer mappings for IDs
user_ids = interactions_df["user_id"].unique()
product_ids = interactions_df["product_id"].unique()

user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
product_to_idx = {pid: i for i, pid in enumerate(product_ids)}
idx_to_product = {i: pid for pid, i in product_to_idx.items()}

# 2. Map DataFrame columns to matrix indices
rows = interactions_df["user_id"].map(user_to_idx)
cols = interactions_df["product_id"].map(product_to_idx)
data = interactions_df["weight"]

# 3. Build the Sparse User-Item Matrix
# Shape: (Num_Users, Num_Products)
user_item_matrix = sparse.csr_matrix(
    (data, (rows, cols)), shape=(len(user_ids), len(product_ids))
)

print(
    f"Sparse matrix built: {user_item_matrix.shape[0]} users × {user_item_matrix.shape[1]} products."
)

# 4. Train iALS (implicit library expects Item x User matrix, so we transpose it)
model = AlternatingLeastSquares(
    factors=32, regularization=0.1, iterations=15, random_state=42
)
model.fit(user_item_matrix.T.tocsr())

# 5. Extract Frozen Embeddings
user_embeddings = model.user_factors  # Shape: (Num_Users, 32)
item_embeddings = model.item_factors  # Shape: (Num_Products, 32)

print("iALS model trained successfully! Frozen embeddings are ready for Stage 1.")
import os
import pickle
from pathlib import Path

# Create a models directory
os.makedirs("models", exist_ok=True)

# 1. Save the iALS model using implicit's built-in method
model.save("models/ials_model.npz")

# 2. Save your user and product mappings using pickle
mappings = {
    "user_to_idx": user_to_idx,
    "product_to_idx": product_to_idx,
    "idx_to_product": idx_to_product,
}

with open("models/mappings.pkl", "wb") as f:
    pickle.dump(mappings, f)

print("Model and mappings successfully saved to 'models/' directory!")
