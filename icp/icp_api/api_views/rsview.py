from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from icp_api.models import Movie
from icp_api.serializers import MovieSerializer  # Assuming this is your serializer
from icp_api.init_recommender import cf_recommender
from rest_framework import generics

class RecommendedMoviesView(generics.ListAPIView):
    serializer_class = MovieSerializer
    permission_classes = [IsAuthenticated]


    def get(self, request):
        user_id = request.user.id
        recommendations = cf_recommender.get_movie_recommendations(user_id)
        movie_recommendations = []

        for recommendation in recommendations:
            # Fetch the movie using movie_id from the recommendation
            try:
                movie = Movie.objects.get(pk=recommendation["movie_id"])
                # Serialize the movie
                serialized_movie = MovieSerializer(movie).data
                # Add the predicted rating to the serialized data
                serialized_movie['predicted_rating'] = float(recommendation['predicted_rating'])
                # Append the result to the movie_recommendations list
                movie_recommendations.append(serialized_movie)
            except Movie.DoesNotExist:
                # Handle cases where the movie might not exist
                continue  # or log the error, etc.

        return JsonResponse(movie_recommendations, safe=False)
