from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from icp_api.models import Rating
from icp_api.serializers import RatingSerializer
from rest_framework.permissions import IsAuthenticated
from django.core.exceptions import ObjectDoesNotExist
from icp_api.models import MovieSlug

class CreateRatingView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        data = request.data.copy()
        movie_slug = data.get('movie_slug')  # Assume 'movie_slug' is passed in the request

        # Attempt to find the Movie instance associated with the given slug
        try:
            movie_slug_instance = MovieSlug.objects.get(movie_slug=movie_slug)
            movie = movie_slug_instance.movie_id  # Directly use the Movie instance associated with the slug
        except ObjectDoesNotExist:
            # If either the slug or the movie does not exist, return an error
            return Response({'message': 'Movie not found.'}, status=status.HTTP_404_NOT_FOUND)

        # Now that we have the movie, replace the slug with the movie ID in the data for serialization
        data['movie'] = movie.movie_id
        data['user'] = request.user.id

        # Check if the user has already rated this movie
        if Rating.objects.filter(user=request.user, movie=movie).exists():
            return Response({'message': 'User has already rated this movie.'}, status=status.HTTP_400_BAD_REQUEST)

        if('rating' not in data):
            return Response({'message': 'Rating is required.'}, status=status.HTTP_400_BAD_REQUEST)

        rating = float(data.get('rating'))
        
        if(rating < 0.5 or rating > 5):
            return Response({'message': 'Rating must be between 0.5 and 5.'}, status=status.HTTP_400_BAD_REQUEST)

        #check in 0.5 increments
        if(rating % 0.5 != 0):
            return Response({'message': 'Rating must be in 0.5 increments.'}, status=status.HTTP_400_BAD_REQUEST)

        serializer = RatingSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
