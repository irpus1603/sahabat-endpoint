from django.urls import path
from . import views

urlpatterns = [
    # Authentication
    path('login/', views.custom_login, name='login'),
    path('logout/', views.custom_logout, name='logout'),
    
    # Main dashboard
    path('', views.dashboard, name='dashboard'),
    
    # Camera management
    path('cameras/', views.camera_settings, name='camera_settings'),
    path('cameras/add/', views.camera_add, name='camera_add'),
    path('cameras/edit/<int:camera_id>/', views.camera_edit, name='camera_edit'),
    path('cameras/delete/<int:camera_id>/', views.camera_delete, name='camera_delete'),
    
    # Snapshots and detections
    path('snapshots/', views.snapshots, name='snapshots'),
    path('detections/', views.detections, name='detections'),
    path('aicamera/', views.aicamera_redirect, name='aicamera'),
    path('aicamera/all/', views.aicamera, name='all_ai'),
    #detection using MPX
    path('detection/security_intrusion/<int:camera_id>/', views.detection_security_intrusion, name='detection_security_intrusion'),
    path('detection/object_detection/<int:camera_id>/', views.detection_object, name='object_detection'),
    path('detection/detection_object_video_source/<int:camera_id>/', views.detection_object_video_source, name='detection_object_video_source'),
    path('detection/ppe_detection/<int:camera_id>/', views.detection_ppe, name='ppe_detection'),
    path('detection/obj_detection_yoloe/<int:camera_id>/', views.obj_det_yoloe, name='obj_detection_yoloe'),

    #Detection using GPU/CPU
    path('detection/det_obj_cpu/<int:camera_id>/', views.det_obj_cpu, name='det_obj_cpu'),
    path('detection/ppe_det_cpu/<int:camera_id>/', views.ppe_det_cpu, name='ppe_det_cpu'),
    path('detection/obj_det_cpu/<int:camera_id>/', views.obj_det_cpu, name='obj_det_cpu'),
    path('detection/obj_det_onnx/<int:camera_id>/', views.obj_det_onnx, name='obj_det_onnx'),

    # Alerts
    path('alerts/', views.alerts, name='alerts'),
    path('alerts/acknowledge/<int:alert_id>/', views.alert_acknowledge, name='alert_acknowledge'),
    
    # System configuration
    path('config/', views.system_config, name='system_config'),
    
    # API endpoints
    path('api/system-status/', views.api_system_status, name='api_system_status'),
    path('api/camera/<int:camera_id>/details/', views.api_camera_details, name='api_camera_details'),
    path('api/camera/<int:camera_id>/stream-info/', views.api_camera_stream_info, name='api_camera_stream_info'),
    path('api/camera/<int:camera_id>/test-connection/', views.api_camera_test_connection, name='api_camera_test_connection'),
    path('api/camera/<int:camera_id>/health/', views.api_camera_health, name='api_camera_health'),
    path('api/camera/<int:camera_id>/hls-check/', views.api_camera_hls_check, name='api_camera_hls_check'),
    path('api/camera/<int:camera_id>/capture-frame/', views.api_camera_capture_frame, name='api_camera_capture_frame'),
    path('api/camera/<int:camera_id>/generate-thumbnail/', views.api_generate_thumbnail, name='api_generate_thumbnail'),
    path('api/camera/<int:camera_id>/stream/hls/', views.api_camera_stream_hls, name='api_camera_stream_hls'),
    path('camera/<int:camera_id>/live/', views.camera_live_stream, name='camera_live_stream'),
    path('api/alerts/acknowledge-all/', views.api_alerts_acknowledge_all, name='api_alerts_acknowledge_all'),
]