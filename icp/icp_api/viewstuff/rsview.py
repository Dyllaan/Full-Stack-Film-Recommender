from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from icp_api.models import Movie
from icp_api.init_recommender import cf_recommender
<<<<<<< Updated upstream
=======
from icp_api.serializers import MovieSerializer
>>>>>>> Stashed changes

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def recommend_movies(request):
    user_id = request.user.id
    recommendations = cf_recommender.get_movie_recommendations(user_id)
<<<<<<< Updated upstream

    # Fetch movie titles for the recommended IDs
    movie_ids = [rec['movie_id'] for rec in recommendations]
    movies = Movie.objects.filter(movie_id__in=movie_ids).values('movie_id', 'movie_title')
    movie_title_map = {movie['movie_id']: movie['movie_title'] for movie in movies}

    # Add movie titles to recommendations
    for recommendation in recommendations:
        recommendation['predicted_rating'] = float(recommendation['predicted_rating'])
        recommendation['title'] = movie_title_map.get(recommendation['movie_id'], "Title not found")

    return JsonResponse(recommendations, safe=False)

=======

    print(recommendations)

    return JsonResponse(recommend_movies, safe=False)
>>>>>>> Stashed changes
