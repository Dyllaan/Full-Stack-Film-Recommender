from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated

class HybridView(APIView):
    permission_classes = [IsAuthenticated]
        
    def get(self, request):
        return Response("Hybrid Recommender API", status=status.HTTP_200_OK)
