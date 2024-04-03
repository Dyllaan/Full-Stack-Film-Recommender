from rest_framework.response import Response
from rest_framework import status
from icp_api.models import Movie, MovieSlug
from icp_api.serializers import MovieSerializer
from rest_framework import generics

class GetMovieBySlug(generics.RetrieveAPIView):
        
    def get_movie_object(self, slug):
        try:
            # Assuming movie_slug field uniquely identifies a MovieSlug instance
            movie_slug = MovieSlug.objects.get(movie_slug=slug)
            # Directly return the Movie instance associated with the slug
            return movie_slug.movie_id
        except (MovieSlug.DoesNotExist, Movie.DoesNotExist):
            return None

    def get(self, request, slug, format=None):
        movie = self.get_movie_object(slug)
        if movie is None:
            return Response(status=status.HTTP_404_NOT_FOUND)
        
        serializer = MovieSerializer(movie)
        return Response(serializer.data)