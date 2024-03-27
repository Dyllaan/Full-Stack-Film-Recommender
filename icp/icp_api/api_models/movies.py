from django.db import models

class Movie(models.Model):
    # Assuming the existing table has columns 'movie_id', 'title', and 'genre'
    movie_id = models.AutoField(primary_key=True, db_column='movie_id')
    movie_title = models.CharField(max_length=200, db_column='movie_title')
    movie_genres = models.CharField(max_length=100, db_column='movie_genres')
    release_year = models.IntegerField(db_column='release_year')

    class Meta:
        db_table = 'movies'  # Name of the existing table

    def __str__(self):
        return self.movie_title