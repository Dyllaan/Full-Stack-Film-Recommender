from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import time
from icp_api.models import Rating
from icp_api.serializers import RatingSerializer
from rest_framework.permissions import IsAuthenticated

class CreateRatingView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request, *args, **kwargs):
        data = request.data.copy()
        data['user'] = request.user.id
        # Check if the user has already rated this film
        if Rating.objects.filter(user=request.user, movie_id=data['movie']).exists():
            return Response({'message': 'User has already rated this movie.'}, status=status.HTTP_400_BAD_REQUEST)
        
        serializer = RatingSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
