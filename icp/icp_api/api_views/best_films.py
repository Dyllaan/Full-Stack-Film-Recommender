from django.http import JsonResponse
from rest_framework.views import APIView
from icp_api.models import Movie, Rating
from django.db.models import Avg
import random
from icp_api.api_serializers.movieserializer import MovieSerializer

class BestFilmsView(APIView):

    def get(self, request):
        # Aggregate to find the average rating for each movie, ordering by avg_rating
        top_ratings = Rating.objects.values('movie_id').annotate(
            avg_rating=Avg('rating')
        ).order_by('-avg_rating')[:1000]  # 1000 ensures that the user is given a random selection of top-rated movies

        movie_list = list(top_ratings)
        # Shuffle it so it appears fresh to the user
        random.shuffle(movie_list)
        # Get the top 10 movies
        top_ratings = movie_list[:10]

        if top_ratings:
            movie_ids = [rating['movie_id'] for rating in top_ratings]
            movies_qs = Movie.objects.filter(movie_id__in=movie_ids)
            data = MovieSerializer(movies_qs, many=True).data
        else:
            data = {'message': 'No films found!'}

        return JsonResponse(data, safe=False)  # Use safe=False for objects serialization
