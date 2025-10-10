from urllib.parse import urlparse
from django.urls import path
from . import views


urlpatterns = [
    path('', views.list_people_counting_camera, name='list_people_counting_camera'),  # List all cameras with People Counting capabilities
    path('count/page/<int:camera_id>/', views.people_counting_detection_page, name='people_counting_detection_page'),  # View specific camera details
    path('count/<int:camera_id>/', views.people_counting_detection, name='people_counting_detection'),  
    
    # API endpoints
    path('api/test/', views.api_test, name='api_test'),
    path('api/events/<int:camera_id>/', views.api_events, name='api_events'),
    path('api/summary/<int:camera_id>/', views.api_summary, name='api_summary'),
    path('api/daily-traffic/<int:camera_id>/', views.api_daily_traffic, name='api_daily_traffic'),
    path('api/hourly-distribution/<int:camera_id>/', views.api_hourly_distribution, name='api_hourly_distribution'),
]