from django.db import models
from .apiusermodel import ApiUser
from .movies import Movie
from django_unixdatetimefield import UnixDateTimeField

class Rating(models.Model):
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    user = models.ForeignKey(ApiUser, on_delete=models.CASCADE)
    rating = models.DecimalField(max_digits=3, decimal_places=2)  # e.g., allows ratings like 8.99
    created_at = UnixDateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.movie} - {self.rating}"
