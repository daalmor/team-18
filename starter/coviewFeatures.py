from datetime import datetime
from collections import defaultdict
import math

def extract_coview_features_from_events(candidate_id, session_events, coview_dict, half_life_seconds=300):
    """
    Extracts coview features for a GBM model from a list of timestamped session events,
    including an explicit binary flag indicating if a coview relationship exists.
    """
    if not session_events:
        return {
            'coview_score': 0.0,
            'coview_frequency': 0,
            'coview_best_rank': 999,
            'coview_triggered_by_last_view': 0,
            'coview_avg_rank': 999.0,
            'time_since_last_trigger_sec': 999999.0,
            'has_coview_signal': 0  # <--- Added flag
        }

    sorted_events = sorted(session_events, key=lambda x: x['ts'])
    recent_views = [event['product_id'] for event in sorted_events]
    viewed_set = set(recent_views)
    
    if candidate_id in viewed_set:
        return {
            'coview_score': 0.0,
            'coview_frequency': 0,
            'coview_best_rank': 999,
            'coview_triggered_by_last_view': 0,
            'coview_avg_rank': 999.0,
            'time_since_last_trigger_sec': 999999.0,
            'has_coview_signal': 0  # <--- Added flag
        }

    last_event_ts = datetime.fromisoformat(sorted_events[-1]['ts'])
    
    total_score = 0.0
    frequency = 0
    ranks = []
    triggered_by_last = 0
    trigger_times = []
    
    for idx, event in enumerate(sorted_events):
        viewed_id = event['product_id']
        event_ts = datetime.fromisoformat(event['ts'])
        
        if viewed_id not in coview_dict:
            continue
            
        coview_list = coview_dict[viewed_id]
        
        if candidate_id in coview_list:
            rank = coview_list.index(candidate_id)
            position_score = 1.0 / (rank + 1)
            
            delta_seconds = (last_event_ts - event_ts).total_seconds()
            recency_weight = math.pow(0.5, delta_seconds / half_life_seconds)
            
            total_score += recency_weight * position_score
            frequency += 1
            ranks.append(rank + 1)
            trigger_times.append(delta_seconds)
            
            if idx == len(sorted_events) - 1:
                triggered_by_last = 1
                
    best_rank = min(ranks) if ranks else 999
    avg_rank = sum(ranks) / len(ranks) if ranks else 999.0
    min_time_since_trigger = min(trigger_times) if trigger_times else 999999.0
    
    # Set to 1 if the candidate appeared in at least one coview list, else 0
    has_signal = 1 if frequency > 0 else 0
    
    return {
        'coview_score': total_score,
        'coview_frequency': frequency,
        'coview_best_rank': best_rank,
        'coview_triggered_by_last_view': triggered_by_last,
        'coview_avg_rank': avg_rank,
        'time_since_last_trigger_sec': min_time_since_trigger,
        'has_coview_signal': has_signal
    }