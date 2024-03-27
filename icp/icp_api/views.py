from django.contrib.auth.models import Group
from icp_api.models import ApiUser
from rest_framework import viewsets
from rest_framework import permissions
from rest_framework.pagination import PageNumberPagination
from .api_models.movies import Movie
from .api_models.ratingmodel import Rating
from icp_api.serializers import UserSerializer, GroupSerializer, MovieSerializer, RatingSerializer
from rest_framework import generics
from django.db.models import Q

class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = ApiUser.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    permission_classes = [permissions.IsAuthenticated]


class MovieList(generics.ListAPIView):
    queryset = Movie.objects.all()
    serializer_class = MovieSerializer


    def get_queryset(self):
        """
        Optionally restricts the returned movies to those that match
        a search query.
        """
        queryset = Movie.objects.all()
        search = self.request.query_params.get('search', None)
        limit = self.request.query_params.get('limit', None)
        if search is not None:
            queryset = queryset.filter(
                Q(movie_title__icontains=search)

            )
        return queryset

class MovieDetail(generics.RetrieveAPIView):
    queryset = Movie.objects.all()
    serializer_class = MovieSerializer
    lookup_field = 'movie_id'

class RatingList(generics.ListCreateAPIView):
    queryset = Rating.objects.all()
    serializer_class = RatingSerializer
    def get_queryset(self):
        user = self.request.user
        return Rating.objects.filter(user=user)

class RatingDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Rating.objects.all()
    serializer_class = RatingSerializer
    