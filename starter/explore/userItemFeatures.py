import pickle
import numpy as np
from implicit.als import AlternatingLeastSquares

# 1. Load the saved factors from the .npz file
print("Loading saved model factors...")
npz_data = np.load("../../models/ials_model.npz")
user_factors = npz_data["user_factors"]
item_factors = npz_data["item_factors"]

# 2. Initialize the model and restore the factors
# (We pass the number of factors based on the shape of the loaded array)
model = AlternatingLeastSquares(factors=user_factors.shape[1])
model.user_factors = user_factors
model.item_factors = item_factors

# 3. Load mappings
with open("../../models/mappings.pkl", "rb") as f:
    mappings = pickle.load(f)

user_to_idx = mappings["user_to_idx"]
product_to_idx = mappings["product_to_idx"]
idx_to_product = mappings["idx_to_product"]

def user_item_features(sample_user_id):
    if sample_user_id in user_to_idx:
        u_idx = user_to_idx[sample_user_id]
        
        # implicit's recommend method will now work using the loaded factors
        item_indices, scores = model.recommend(
            userid=u_idx, 
            user_items=None, 
            N=10,
            filter_already_liked_items=True
        )
        
        # Map indices back to product IDs
        top_products = [idx_to_product[idx] for idx in item_indices]
        
        print("\nTop 10 Recommendations:")
        for rank, (prod_id, score) in enumerate(zip(top_products, scores), 1):
            print(f"{rank}. Product ID: {prod_id} (Score: {score:.4f})")
    else:
        print(f"User {sample_user_id} not found in training mappings (Cold-start case).")