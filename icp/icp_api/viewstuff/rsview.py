from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response  # Import the Response class
from icp_api.models import Movie
from icp_api.init_recommender import get_model
from icp_api.serializers import MovieSerializer

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def recommend_movies(request):
    user_id = request.user.id
    recommendations = get_model().get_movie_recommendations(user_id)

    # Create a dictionary of prediction ratings keyed by movie_id
    prediction_ratings = {str(rec['movie_id']): rec['predicted_rating'] for rec in recommendations}

    movie_ids = [rec['movie_id'] for rec in recommendations]
    movies = Movie.objects.filter(movie_id__in=movie_ids)

    # Pass the prediction ratings in the serializer context
    serializer = MovieSerializer(movies, many=True, context={'prediction_ratings': prediction_ratings})

    return JsonResponse(serializer.data, safe=False)
