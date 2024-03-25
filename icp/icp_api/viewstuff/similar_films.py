# views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from icp_api.api_models.movie_recommender import MovieRecommender  # Adjust the import path as necessary
from icp_api.models import Movie  # Ensure you have this import
from icp_api.serializers import MovieSerializer

class SimilarMoviesView(APIView):
    def get(self, request, movie_id, format=None):
        # Instantiate the recommender
        recommender = MovieRecommender()
        
        # Convert movie_id to movie title
        # This assumes you have a method to get the movie title by its ID
        try:
            movie_title = Movie.objects.get(movie_id=movie_id).title
        except Movie.DoesNotExist:
            return Response({"error": "Movie not found"}, status=404)
        
        # Get content-based recommendations based on the movie title
        similar_titles = recommender.get_content_based_recommendations(movie_title)
        
        # Fetch the movie objects for the recommended titles
        similar_movies = Movie.objects.filter(title__in=similar_titles)
        
        # Serialize the movie data
        serializer = MovieSerializer(similar_movies, many=True)
        
        # Return the serialized data
        return Response(serializer.data)
