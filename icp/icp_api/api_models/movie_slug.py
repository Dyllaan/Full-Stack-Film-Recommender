from django.db import models
from .movies import Movie

class MovieSlug(models.Model):
    
    slug_id = models.AutoField(primary_key=True, db_column='slug_id')
    movie_id = models.OneToOneField(Movie, on_delete=models.CASCADE, db_column='movie_id')
    movie_slug = models.SlugField(max_length=255, unique=True)

    class Meta:
        db_table = 'movie_slugs'  # Name of the existing table

    def __str__(self):
        return "Movie: " + self.movie_title + " - Slug: " + self.movie_slug + " - ID: " + str(self.movie_id)
    
    def get_movie_title(self):
        return self.movie_id.movie_title