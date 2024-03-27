from icp_api.rs_models.collaborative_based_recommender import CFRecommender
from icp_api.rs_models.content_based_recommender import CBFRecommender

# Initialize the recommender system
cf_recommender = CFRecommender()
cf_recommender.startup()

# Initialise cbf

cbf_recommender = CBFRecommender()
cbf_recommender.startup()