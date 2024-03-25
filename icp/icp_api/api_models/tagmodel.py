from django.db import models
from .movies import Movie
from .apiusermodel import ApiUser

class Tag(models.Model):
    user_id = models.ForeignKey(ApiUser, on_delete=models.CASCADE)
    movie_id = models.ForeignKey(Movie, on_delete=models.CASCADE, db_column='movie_id', null=True)
    tag = models.CharField(max_length=255)  # Assuming tags won't exceed 255 characters
    timestamp = models.BigIntegerField()  # Storing the timestamp as a BigInteger

    def __str__(self):
        return f"{self.tag} on {self.movie_id} by {self.user_id}"
