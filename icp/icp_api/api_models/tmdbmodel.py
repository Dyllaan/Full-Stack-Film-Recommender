from django.db import models
from .movies import Movie

class TMDb(models.Model):
    tmdb_id = models.IntegerField(primary_key=True)
    movie_id = models.ForeignKey(Movie, on_delete=models.CASCADE, db_column='movie_id', null=True)

    class Meta:
        db_table = 'tmdb'

    def __str__(self):
        movie_title = self.movie_id.movie_title
        return f"TMDB {self.tmdb_id}: Movie {movie_title}"