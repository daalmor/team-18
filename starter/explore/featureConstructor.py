def merge_dictionaries(dict1, dict2):
    """
    Merges two dictionaries. If there are overlapping keys, 
    values from dict2 will take precedence over dict1.
    """
    return {**dict1, **dict2}

from coviewFeatures import extract_coview_features_from_events

def constructFeatures(
        product_id,
        brand,
        category,
        created_at,
        description,
        is_active,
        price_eur,
        subcategory,
        ts, user_id, is_new_user, context, recent_views,
        coview_dict):

    X = {"is_new_user"   : is_new_user,
         "device"        : context.device,
         "region"        : context.region,
         "page"          : context.page}

    merge_dictionaries(extract_coview_features_from_events(product_id, recent_views, coview_dict))
