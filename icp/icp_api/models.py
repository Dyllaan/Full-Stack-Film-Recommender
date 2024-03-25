from django.db import models
from django.contrib.auth.models import AbstractUser
from .api_models.movies import Movie
from .api_models.apiusermodel import ApiUser
from .api_models.ratingmodel import Rating
from .api_models.tmdb_posters_model import TMDbPoster
from .api_models.tmdbmodel import TMDb
from .api_models.tagmodel import Tag