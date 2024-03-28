from celery import shared_task
from icp_api.rs_models.collaborative_based_recommender import CFRecommender

@shared_task
def update_recommender_system():
    recommender = CFRecommender()
    recommender.startup()  
