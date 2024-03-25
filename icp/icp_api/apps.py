from django.apps import AppConfig
from icp_api.api_models.movie_recommender import MovieRecommender
from django.db.models.signals import post_migrate
from django.dispatch import receiver

class IcpApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'icp_api'

    def ready(self):
        post_migrate.connect(run_recommender_startup, sender=self)

@receiver(post_migrate)
def run_recommender_startup(sender, **kwargs):
    if sender.name == 'icp_api':
        recommender = MovieRecommender()
        recommender.startup()