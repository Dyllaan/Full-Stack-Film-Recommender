from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from icp_api.models import Movie, MovieSlug, Rating  # Assuming UserRating is your user rating model
from icp_api.serializers import MovieSerializer
from icp_api.init_recommender import cbf_recommender
from rest_framework import status
import random

class SimilarMoviesView(APIView):

    def get(self, request, movie_slug=None):
        basis_movie_info = {}

        if movie_slug:
            # Attempt to find the MovieSlug instance associated with the given slug
            movie_slug_instance = get_object_or_404(MovieSlug, movie_slug=movie_slug)
            movie = movie_slug_instance.movie_id
            movie_title = movie.movie_title
            # Include movie slug and title in the response to indicate the basis
            basis_movie_info = {'movie_slug': movie_slug, 'movie_title': movie_title}
        else:
            # Select a random movie from the user's ratings if no slug is provided
            user = request.user  # Ensure you have access to the request user
            user_ratings = Rating.objects.filter(user=user)
            if user_ratings.exists():
                random_rating = random.choice(user_ratings)  # Select a random rating
                movie = random_rating.movie
                movie_title = movie.movie_title
                basis_movie_info = {'movie_title': movie_title}
            else:
                return Response({'message': 'No ratings found for user to base recommendations on.'}, status=status.HTTP_404_NOT_FOUND)

        # Get content-based recommendations based on the movie title
        similar_titles = cbf_recommender.get_content_based_recommendations(movie_title)

        # Fetch the movie objects for the recommended titles
        similar_movies = Movie.objects.filter(movie_title__in=similar_titles)

        # Serialize the movie data
        serializer = MovieSerializer(similar_movies, many=True)

        serialized_basis_movie = MovieSerializer(movie).data

        # Return the serialized data along with the basis movie info
        response_data = {
            'similar_to': serialized_basis_movie,
            'recommendations': serializer.data
        }
        return Response(response_data)
