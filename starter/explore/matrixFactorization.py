def extract_coview_features(candidate_id, recent_views, coview_dict):
    """
    Extracts session-based coview features for a specific candidate product
    to be used as features in a GBM model.
    
    Args:
        candidate_id (str): The candidate product being evaluated.
        recent_views (list): Product IDs viewed by the user in the session (oldest to newest).
        coview_dict (dict): Dict mapping productID -> list of top 20 co-viewed product IDs.
        
    Returns:
        dict: Feature dictionary for the GBM.
    """
    num_views = len(recent_views)
    viewed_set = set(recent_views)
    
    # Default features if the user already viewed the candidate or no data exists
    if candidate_id in viewed_set or num_views == 0:
        return {
            'coview_score': 0.0,
            'coview_frequency': 0,
            'coview_best_rank': 999,      # High number means worst possible rank
            'coview_triggered_by_last_view': 0,
            'coview_avg_rank': 999.0
        }
        
    total_score = 0.0
    frequency = 0
    ranks = []
    triggered_by_last = 0
    
    for idx, viewed_id in enumerate(recent_views):
        # Recency weight (older items get less weight, last item gets 1.0)
        recency_weight = (idx + 1) / num_views  
        
        if viewed_id not in coview_dict:
            continue
            
        coview_list = coview_dict[viewed_id]
        
        if candidate_id in coview_list:
            rank = coview_list.index(candidate_id) # 0-based index
            position_score = 1.0 / (rank + 1)
            
            total_score += recency_weight * position_score
            frequency += 1
            ranks.append(rank + 1) # Store 1-based rank (1 is top)
            
            # Check if this candidate was triggered by the very last view in the session
            if idx == num_views - 1:
                triggered_by_last = 1
                
    # Calculate aggregates
    best_rank = min(ranks) if ranks else 999
    avg_rank = sum(ranks) / len(ranks) if ranks else 999.0
    
    return {
        'coview_score': total_score,
        'coview_frequency': frequency,
        'coview_best_rank': best_rank,
        'coview_triggered_by_last_view': triggered_by_last,
        'coview_avg_rank': avg_rank
    }