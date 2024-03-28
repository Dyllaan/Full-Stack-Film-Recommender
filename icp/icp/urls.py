from django.urls import include, path
from rest_framework import routers
from icp_api import views
from icp_api.views import MovieList, MovieDetail, RatingList, RatingDetail
from rest_framework_simplejwt.views import TokenRefreshView
from icp_api.viewstuff.registerview import RegisterView
from icp_api.viewstuff.currentuser import CurrentUserView
from icp_api.auth.usertokenview import UserTokenView  # Import your custom view
from icp_api.viewstuff.rsview import recommend_movies
from icp_api.viewstuff.fakeusers import GenerateFakeUsersView
from icp_api.viewstuff.similar_films import SimilarMoviesView
from icp_api.viewstuff.hybrid_view import HybridView
from icp_api.viewstuff.post.CreateRating import CreateRatingView
from icp_api.viewstuff.get_movie_by_slug import GetMovieBySlug
router = routers.DefaultRouter()

urlpatterns = [
    path('movies', MovieList.as_view(), name='movies-list'),
    path('movie/<int:movie_id>', MovieDetail.as_view(), name='movie-detail'),
    path('movie/<str:slug>', GetMovieBySlug.as_view(), name='movie-by-slug'),
    path('ratings', RatingList.as_view(), name='rating-list'),
    path('ratings/<int:pk>', RatingDetail.as_view(), name='rating-detail'),
    path('ratings/create', CreateRatingView.as_view(), name='create-rating'),
    # authorizing the user
    path('user/login', UserTokenView.as_view(), name='custom_token_obtain_pair'),  # Use your custom view
    path('user/refresh', TokenRefreshView.as_view(), name='token_refresh'),
    path('user/register', RegisterView.as_view(), name='register'),
    path('user', CurrentUserView.as_view(), name='user'),
    path('recommendations', recommend_movies, name='recommendations'),
    path('similar_movies/<int:movie_id>', SimilarMoviesView.as_view(), name='similar_movies'),
    path('hybrid_recommendations', HybridView.as_view(), name='hybrid_recommendations'),
]
