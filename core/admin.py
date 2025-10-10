from django.contrib import admin
from .models import Aimodule, Camera, Detection, Alert, PPEChecks, PeopleCountingEvent, SystemConfig, intrusionChecks

@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ['name','thumbnail','camera_source', 'rtsp_url', 'is_active', 'created_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['name']
    readonly_fields = ['created_at']

@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ['camera', 'description', 'timestamp']
    list_filter = ['camera', 'timestamp']
    search_fields = ['description']
    readonly_fields = ['timestamp']

@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ['camera', 'severity', 'message', 'is_acknowledged', 'created_at']
    list_filter = ['severity', 'is_acknowledged', 'camera', 'created_at']
    search_fields = ['message']
    readonly_fields = ['created_at']

@admin.register(SystemConfig)
class SystemConfigAdmin(admin.ModelAdmin):
    list_display = ['key', 'value', 'category']
    list_filter = ['category']
    search_fields = ['key', 'value']


@admin.register(Aimodule)
class AimoduleAdmin(admin.ModelAdmin):
    list_display = ['name', 'module','description']
    search_fields = ['name','module']

@admin.register(PPEChecks)
class PPEChecksAdmin(admin.ModelAdmin):
    list_display = ['camera', 'Detectionid', 'trackid', 'has_helmet', 'has_vest', 'has_goggles', 'has_gloves', 'has_shoes']
    list_filter = ['camera', 'has_helmet', 'has_vest', 'has_goggles', 'has_gloves', 'has_shoes']
    search_fields = ['trackid']

@admin.register(intrusionChecks)
class intrusionChecksAdmin(admin.ModelAdmin):
    list_display = ['camera', 'Detectionid', 'trackid', 'intrusion', 'intrusion_type', 'severity', 'created_at']
    list_filter = ['camera', 'intrusion', 'intrusion_type', 'severity', 'created_at']
    search_fields = ['trackid']

@admin.register(PeopleCountingEvent)
class PeopleCountingEventAdmin(admin.ModelAdmin):
    list_display = ['camera', 'frameid', 'timestamp']
    list_filter = ['camera', 'timestamp']
    search_fields = ['frameid']
    readonly_fields = ['timestamp']