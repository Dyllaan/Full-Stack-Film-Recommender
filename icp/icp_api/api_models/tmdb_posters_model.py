from django.db import models
from .movies import Movie
from .tmdbmodel import TMDb

class TMDbPoster(models.Model):
    poster_id = models.AutoField(primary_key=True)
    tmdb_id = models.ForeignKey(TMDb, on_delete=models.CASCADE, db_column='tmdb_id', null=True)
    poster_path = models.CharField(max_length=255, null=False)

    class Meta:
        db_table = 'tmdb_posters'

    def __str__(self):
        movie_title = self.movie_id.movie_title
        return f"Poster {self.poster_id}: Movie: {self.movie_id},  IMDB: {self.imdb_id}, TMDB: {self.tmdb_id}, Poster Path: {self.poster_path}"
