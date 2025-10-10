from urllib.parse import urlparse
from django.urls import path
from . import views


urlpatterns = [
    path('', views.list_intrusion_camera, name='list_intrusion_camera'),  # List all cameras with security intrusion
    path('detection/page/<int:camera_id>/', views.intrusion_detection_page, name='intrusion_detection_page'),  # View specific camera details
    path('detection/<int:camera_id>/', views.intrusion_detection, name='intrusion_detection'),  
    path('reports/', views.intrusion_report, name='intrusion_report'),  # Comprehensive security reports
    
    # API endpoints for chart data
    path('api/violations/<int:camera_id>/', views.api_intrusion_violations, name='api_intrusion_violations'),
    path('api/summary/<int:camera_id>/', views.api_intrusion_summary, name='api_intrusion_summary'),
    path('api/severity-breakdown/<int:camera_id>/', views.api_intrusion_severity_breakdown, name='api_intrusion_severity_breakdown'),
    path('api/type-breakdown/<int:camera_id>/', views.api_intrusion_type_breakdown, name='api_intrusion_type_breakdown'),
    path('api/latest-incidents/<int:camera_id>/', views.api_latest_incidents, name='api_latest_incidents'),
    path('api/test/', views.api_test, name='api_intrusion_test'),
]
