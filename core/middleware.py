from django.shortcuts import redirect
from django.urls import reverse
from django.conf import settings


class LoginRequiredMiddleware:
    """
    Middleware that requires a user to be authenticated to view any page
    except for the admin login page and static/media files.
    """
    def __init__(self, get_response):
        self.get_response = get_response
        # One-time configuration and initialization.

    def __call__(self, request):
        # Code to be executed for each request before
        # the view (and later middleware) are called.

        # URLs that don't require authentication
        exempt_urls = [
            '/login/',
            '/admin/login/',
            '/admin/',
            '/static/',
            '/media/',
        ]
        
        # Check if the current path requires authentication
        path = request.path_info
        
        # Allow access to exempt URLs
        if any(path.startswith(url) for url in exempt_urls):
            response = self.get_response(request)
            return response
        
        # Check if user is authenticated
        if not request.user.is_authenticated:
            # Redirect to login page
            return redirect(settings.LOGIN_URL)
        
        response = self.get_response(request)

        # Code to be executed for each request/response after
        # the view is called.

        return response