from django.http import JsonResponse

class CorsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Add headers to all responses
        if request.method == "OPTIONS":
            # If this is a preflight OPTIONS request, return a 200 OK response with the required headers
            response = JsonResponse({"detail": "Options OK"}, status=200)
            response["Access-Control-Allow-Origin"] = "*"
            response["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type, Authorization"
            return response
        else:
            # For all other requests, proceed as normal
            response = self.get_response(request)
            response["Access-Control-Allow-Origin"] = "*"
            response["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type, Authorization"
            return response
