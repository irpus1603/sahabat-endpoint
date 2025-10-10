from django.urls import path
from . import views

urlpatterns = [
    path('', views.list_ppe_camera, name='list_ppe_camera'),
    path('detection/<int:camera_id>/', views.detection_ppe, name='detection_ppe'),
    path('detection/page/<int:camera_id>/', views.ppe_detection_page, name='ppe_detection_page'),
    path('api/violations/<int:camera_id>/', views.api_violation_data, name='api_violation_data'),
    path('api/summary/<int:camera_id>/', views.api_violation_summary, name='api_violation_summary'),
    path('api/equipment-violations/<int:camera_id>/', views.api_equipment_violations, name='api_equipment_violations'),
    path('api/test/', views.api_test, name='api_test'),
    
    # PPE Reports URLs
    path('reports/', views.ppe_reports, name='ppe_reports'),
    path('api/reports/kpi/', views.api_reports_kpi, name='api_reports_kpi'),
    path('api/reports/compliance-trend/', views.api_reports_compliance_trend, name='api_reports_compliance_trend'),
    path('api/reports/violation-types/', views.api_reports_violation_types, name='api_reports_violation_types'),
    path('api/reports/violation-heatmap/', views.api_reports_violation_heatmap, name='api_reports_violation_heatmap'),
    path('api/reports/location-compliance/', views.api_reports_location_compliance, name='api_reports_location_compliance'),
    path('api/reports/trend-analysis/', views.api_reports_trend_analysis, name='api_reports_trend_analysis'),
    path('api/reports/top-violators/', views.api_reports_top_violators, name='api_reports_top_violators'),
    path('api/reports/summary-table/', views.api_reports_summary_table, name='api_reports_summary_table'),
    path('api/reports/export/', views.api_reports_export, name='api_reports_export'),
    path('api/reports/export-table/', views.api_reports_export_table, name='api_reports_export_table'),

]
