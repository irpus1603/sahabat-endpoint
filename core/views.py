import re

from numpy import full
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, permission_required
from django.core.exceptions import PermissionDenied
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.core.paginator import Paginator
from django.db.models import Count, Q
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views.generic import TemplateView
from django.core.cache import cache
import json
from datetime import datetime, timedelta
from .models import Camera, Detection, Alert, SystemConfig, Aimodule
from .mediamtx_update import add_mediamtx_source, replace_mediamtx_source, update_mediamtx_source, remove_mediamtx_source, check_mediamtx_source, check_hls_stream_availability
import logging
import detection.object_detection as obj
import detection.object_detection_video_source as obj_det_vid_source
import detection.ppe_detection as ppe
import detection.security_intrusion as sec
import detection.object_detection_cpu_gpu as detObjCpu
import detection.ppe_detection_cpu as ppe_cpu
import detection.object_detection_cpu as obj_cpu
import detection.object_detection_onnx as obj_onnx
import detection.object_detection_yoloe as obj_yoloe

from core import mediamtx_update


# configure logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import OpenCV, but handle gracefully if not available
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

class OptimizedDashboardView(LoginRequiredMixin, TemplateView):
    """Optimized class-based dashboard view with reduced database queries"""
    template_name = 'core/index.html'
    login_url = '/login/'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get cached dashboard data or compute if not cached
        dashboard_data = cache.get('dashboard_data')
        if dashboard_data is None:
            dashboard_data = self._compute_dashboard_data()
            cache.set('dashboard_data', dashboard_data, 30)  # Cache for 30 seconds
        
        context.update(dashboard_data)
        return context
    
    def _compute_dashboard_data(self):
        """Compute dashboard data with optimized queries and async health checks"""
        today = timezone.now().date()
        
        # Single optimized query for camera statistics using aggregation
        camera_stats = Camera.objects.aggregate(
            total_cameras=Count('id'),
            active_cameras=Count('id', filter=Q(is_active=True)),
            inactive_cameras=Count('id', filter=Q(is_active=False))
        )
        
        # Single optimized query for alert statistics using aggregation
        alert_stats = Alert.objects.aggregate(
            total_alerts=Count('id', filter=Q(is_acknowledged=False)),
            critical_alerts=Count('id', filter=Q(is_acknowledged=False, severity='critical')),
            high_alerts=Count('id', filter=Q(is_acknowledged=False, severity='high')),
            medium_alerts=Count('id', filter=Q(is_acknowledged=False, severity='medium')),
            low_alerts=Count('id', filter=Q(is_acknowledged=False, severity='low'))
        )
        
        # Single query for today's detections
        today_detections = Detection.objects.filter(timestamp__date=today).count()
        
        # Get recent data with select_related for efficiency
        recent_alerts = Alert.objects.select_related('camera', 'detection').order_by('-created_at')[:5]
        recent_detections = Detection.objects.select_related('camera').order_by('-timestamp')[:3]
        
        # Get cameras for live feed (limit to reasonable number)
        all_cameras = Camera.objects.all().order_by('name')[:12]  # Limit to prevent overload
        active_cameras_list = Camera.objects.filter(is_active=True).order_by('name')[:12]
        
        # Calculate system load with async considerations
        active_count = camera_stats['active_cameras']
        # Use cached camera health data if available for more accurate load calculation
        health_cache_keys = [f"camera_health_{cam.id}" for cam in active_cameras_list]
        healthy_cameras = 0
        for key in health_cache_keys:
            health_data = cache.get(key)
            if health_data and health_data.get('status') == 'healthy':
                healthy_cameras += 1
        
        # Base load on active cameras, adjust for actual health status
        base_load = min((active_count * 10), 80) if active_count > 0 else 0
        health_adjustment = min((healthy_cameras * 5), 20) if healthy_cameras > 0 else 0
        system_load = min(base_load + health_adjustment, 100)
        
        # Get system configuration with caching
        config_cache_key = 'system_config'
        system_config = cache.get(config_cache_key)
        if system_config is None:
            system_config = {
                'detection_sensitivity': SystemConfig.get_value('detection_sensitivity', '0.7'),
                'system_name': SystemConfig.get_value('system_name', 'SOCA Edge Surveillance'),
                'mediamtx_ip': SystemConfig.get_value('mediamtx_IP')
            }
            cache.set(config_cache_key, system_config, 300)  # Cache for 5 minutes
        
        return {
            'total_cameras': camera_stats['total_cameras'],
            'active_cameras': camera_stats['active_cameras'],
            'inactive_cameras': camera_stats['inactive_cameras'],
            'total_alerts': alert_stats['total_alerts'],
            'critical_alerts': alert_stats['critical_alerts'],
            'high_alerts': alert_stats['high_alerts'],
            'medium_alerts': alert_stats['medium_alerts'],
            'low_alerts': alert_stats['low_alerts'],
            'today_detections': today_detections,
            'recent_alerts': recent_alerts,
            'recent_detections': recent_detections,
            'all_cameras': all_cameras,
            'active_cameras_list': active_cameras_list,
            'system_load': int(system_load),
            'detection_sensitivity': system_config['detection_sensitivity'],
            'system_name': system_config['system_name'],
            'mediamtx_ip': system_config['mediamtx_ip'],
        }

def custom_login(request):
    """Custom login view"""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        if username and password:
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                next_url = request.POST.get('next') or request.GET.get('next') or '/'
                messages.success(request, f'Welcome back, {user.get_full_name() or user.username}!')
                return redirect(next_url)
            else:
                messages.error(request, 'Invalid username or password.')
        else:
            messages.error(request, 'Please enter both username and password.')
    
    # If user is already authenticated, redirect to dashboard
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    context = {
        'next': request.GET.get('next', ''),
        'page_title': 'Login - SOCA Edge Surveillance'
    }
    return render(request, 'core/login.html', context)

def custom_logout(request):
    """Custom logout view"""
    if request.user.is_authenticated:
        messages.info(request, 'You have been logged out successfully.')
    logout(request)
    return redirect('login')

# Keep the function-based view for backward compatibility
@login_required(login_url='/login/')
def dashboard(request):
    """Backward compatible dashboard view"""
    view = OptimizedDashboardView()
    view.request = request
    context = view.get_context_data()
    return render(request, 'core/index.html', context)

@login_required(login_url='/login/')
def camera_settings(request):
    """Camera management and configuration view"""
    cameras = Camera.objects.all().order_by('name')
    aimodules = Aimodule.objects.filter(is_active=True).order_by('name')
    
    # Calculate camera statistics
    total_cameras = cameras.count()
    active_cameras = cameras.filter(is_active=True).count()
    inactive_cameras = cameras.filter(is_active=False).count()
    
    context = {
        'cameras': cameras,
        'aimodules': aimodules,
        'total_cameras': total_cameras,
        'active_cameras': active_cameras,
        'inactive_cameras': inactive_cameras,
        'streaming_cameras': active_cameras,  # Assuming active cameras are streaming
        'page_title': 'Camera Settings',
        'page_subtitle': 'Manage camera configurations'
    }
    
    return render(request, 'core/camera_settings.html', context)

@permission_required('core.add_camera', raise_exception=True)
def camera_add(request):
    """Add new camera view"""
    if request.method == 'POST':
        name = request.POST.get('name')
        rtsp_url = request.POST.get('rtsp_url')
        username = request.POST.get('username')
        password = request.POST.get('password')
        is_active = request.POST.get('is_active', 'true') == 'true'
        camera_source = request.POST.get('camera_source', 'rtsp')
        
        # Get new location fields
        site_name = request.POST.get('site_name')
        floor = request.POST.get('floor')
        location = request.POST.get('location')
        
        # Handle AI module assignment
        aimodule_id = request.POST.get('aimodule')
        aimodule = None
        if aimodule_id:
            try:
                aimodule = Aimodule.objects.get(id=aimodule_id)
            except Aimodule.DoesNotExist:
                pass
        
        # Handle AI module active status
        is_aimodule_active = request.POST.get('is_aimodule_active', 'false') == 'true'
        
        if name and rtsp_url:
            # Create camera in database
            camera = Camera.objects.create(
                name=name,
                rtsp_url=rtsp_url,
                username=username if username else None,
                password=password if password else None,
                is_active=is_active,
                camera_source=camera_source,
                aimodule=aimodule,
                is_aimodule_active=is_aimodule_active,
                site_name=site_name if site_name else None,
                floor=floor if floor else None,
                location=location if location else None
            )
            
            # Generate thumbnail for new camera if it's active
            if is_active:
                try:
                    generate_and_save_thumbnail(camera)
                    logger.info(f"Thumbnail generated for new camera {camera.name}")
                except Exception as e:
                    logger.warning(f"Failed to generate thumbnail for new camera {camera.name}: {e}")
            
            # Prepare RTSP URL with credentials for MediaMTX
            if username and password:
                if '://' in rtsp_url:
                    protocol, rest = rtsp_url.split('://', 1)
                    full_rtsp_url = f"{protocol}://{username}:{password}@{rest}"
                else:
                    full_rtsp_url = rtsp_url
            else:
                full_rtsp_url = rtsp_url
            
            # Handle MediaMTX configuration
            try: 
                check_camera = check_mediamtx_source(camera.name)
                
                if not check_camera:
                    # Camera doesn't exist in MediaMTX, create it
                    add = add_mediamtx_source(camera.name, full_rtsp_url)
                    if add:
                        logger.info(f"MediaMTX configuration added for {camera.name}")
                else:
                    # Camera exists in MediaMTX, update it
                    update = update_mediamtx_source(camera.name, full_rtsp_url)
                    if update:
                        logger.info(f"MediaMTX configuration updated for {camera.name}")
                                    
            except Exception as e:
                logger.warning(f"Warning: Error updating MediaMTX configuration: {e}")
            
            # Handle AJAX requests
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'message': f'Camera "{name}" has been added successfully.'
                })
            
            # Success message and redirect for regular requests
            messages.success(request, f'Camera "{name}" has been added successfully.')
            return redirect('camera_settings')
        else:
            # Handle AJAX error response
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': False,
                    'message': 'Please provide camera name and RTSP URL.'
                })
            messages.error(request, 'Please provide camera name and RTSP URL.')
    
    return render(request, 'core/camera_add.html', {
        'page_title': 'Add Camera',
        'page_subtitle': 'Configure new camera'
    })

@permission_required('core.change_camera', raise_exception=True)
def camera_edit(request, camera_id):
    """Edit camera configuration view, RTSP camera send to mediamtx server to be loaded, 
       while the connection from dashboard using hls conenction"""
    #check http request 
    camera = get_object_or_404(Camera, id=camera_id)
    original_name = camera.name
    original_rtsp_url = camera.rtsp_url
    original_is_active = camera.is_active
    original_username = camera.username
    original_password = camera.password
    full_rtsp_url = ""


    # handle post request and update to database 
    if request.method == 'POST':
        # Debug logging
        logger.info(f"POST request received - Content-Type: {request.headers.get('Content-Type')}")
        logger.info(f"X-Requested-With: {request.headers.get('X-Requested-With')}")
        logger.info(f"camera status from db: {camera.is_active}")
        logger.info(f"camera status from post: {request.POST.get('is_active')}")
        rtsp_url = request.POST.get('rtsp_url', camera.rtsp_url)
        if rtsp_url[0:4] == 'rtsp':
            protocol, rest = rtsp_url.split('://', 1)
            rtsp_username = request.POST.get('username') if request.POST.get('username') else original_username
            rtsp_password = request.POST.get('password') if request.POST.get('password') else original_password
            full_rtsp_url = f"{protocol}://{rtsp_username}:{rtsp_password}@{rest}" 

        # Handle JSON requests for ROI saving and status toggle
        if (request.headers.get('Content-Type') == 'application/json' or 
            request.headers.get('X-Requested-With') == 'XMLHttpRequest'):
            import json
            try:
                # Try to get JSON data from body, fallback to empty dict if body already consumed
                try:
                    data = json.loads(request.body)
                except:
                    data = {}
                
                if data.get('action') == 'save_roi':
                    # Frontend sends 'roi_data', backend expects 'roi_coordinates'
                    roi_data = data.get('roi_data') or data.get('roi_coordinates')
                    if roi_data:
                        # Save the ROI coordinates
                        camera.set_roi_coordinates(roi_data)
                        
                        # Determine the primary ROI type from the data
                        # If multiple types exist, use the first one or most common one
                        roi_type = 'polygon'  # default
                        if isinstance(roi_data, list) and len(roi_data) > 0:
                            first_roi = roi_data[0]
                            if isinstance(first_roi, dict) and 'type' in first_roi:
                                roi_type = first_roi['type']
                        
                        # Save the ROI type
                        camera.roi_type = roi_type
                        camera.save()
                        
                        print(f"ROI saved - Type: {roi_type}, Data: {roi_data}") 
                        return JsonResponse({
                            'success': True,
                            'message': f'ROI configuration saved successfully (Type: {roi_type})'
                        })
                    else:
                        return JsonResponse({
                            'success': False,
                            'message': 'No ROI data provided'
                        })
                elif data.get('action') == 'toggle_status':
                    status = bool(data.get('is_active'))
                    camera.is_active = data.get('is_active', False)
                    
                    update = update_mediamtx_source(camera.name, full_rtsp_url, status)
                    logger.info(f"MediaMTX configuration updated for {camera.name} status to {status}")

                    camera.save()
                    
                    return JsonResponse({
                        'success': True,
                        'message': f'Camera {"enabled" if camera.is_active else "disabled"} successfully'
                    })
            except json.JSONDecodeError:
                return JsonResponse({
                    'success': False,
                    'message': 'Invalid JSON data'
                })
        
        # Handle regular form data
      
        camera.name = request.POST.get('name', camera.name)
        camera.rtsp_url = request.POST.get('rtsp_url', camera.rtsp_url)
        camera.username = request.POST.get('username', camera.username)
        camera.camera_source = request.POST.get('camera_source', camera.camera_source)
        
        # Update new location fields
        camera.site_name = request.POST.get('site_name', camera.site_name)
        camera.floor = request.POST.get('floor', camera.floor)
        camera.location = request.POST.get('location', camera.location)
        
        password = request.POST.get('password')

        if password:  # Only update password if provided
            camera.password = password
        # Only update is_active if explicitly provided in POST data
        if 'is_active' in request.POST:
            camera.is_active = request.POST.get('is_active') == 'True'

        # Handle AI module assignment
        aimodule_id = request.POST.get('aimodule')
        if aimodule_id:
            try:
                aimodule = Aimodule.objects.get(id=aimodule_id)
                camera.aimodule = aimodule
            except Aimodule.DoesNotExist:
                camera.aimodule = None
        else:
            camera.aimodule = None
        
        # Handle AI module active status
        if 'is_aimodule_active' in request.POST:
            camera.is_aimodule_active = request.POST.get('is_aimodule_active') == 'true' or request.POST.get('is_aimodule_active') == 'on'
        

        # Check MediaMTX camera configuration
        try: 
            check_camera = check_mediamtx_source(camera.name)
            update_status = request.POST.get('is_active')
            print(f"Camera {camera.name} is_active: {request.POST.get('is_active')}")
            # update if camera name, rtsp_url or is_active status has changed
            print(full_rtsp_url)
            if rtsp_url[0:4] == 'rtsp':
                if (original_rtsp_url != request.POST.get('rtsp_url')) or original_name != request.POST.get('name') or request.POST.get('is_active') != original_is_active:
                    if check_camera and camera.name != request.POST.get('name'):
                        # update camera source in MediaMTX
                        update = replace_mediamtx_source(camera.name, request.POST.get('name'), full_rtsp_url)
                        if update:
                            logger.info(f"MediaMTX configuration updated for {camera.name} to {request.POST.get('name')}")

                    elif check_camera and camera.name == request.POST.get('name'): 
                        # If camera exists and name is unchanged, update source only
                        update = update_mediamtx_source(camera.name, full_rtsp_url, update_status)
                        
                        if update:
                            logger.info(f"MediaMTX configuration updated for {camera.name} status to {update_status}")
                    
                    elif not check_camera:
                        # If camera does not exist, create it
                        add = add_mediamtx_source(request.POST.get('name'), full_rtsp_url)
                        if add:
                            logger.info(f"MediaMTX configuration added for {request.POST.get('name')}")
                                        
        except Exception as e:
            logger.warning(f"Warning: Error updating MediaMTX configuration: {e}")
        
        # Handle thumbnail upload
        if 'thumbnail' in request.FILES:
            camera.thumbnail = request.FILES['thumbnail']
        
        camera.save()
        
        # Generate thumbnail if camera is active and no manual thumbnail was uploaded
        if camera.is_active and 'thumbnail' not in request.FILES:
            try:
                generate_and_save_thumbnail(camera)
                logger.info(f"Thumbnail auto-generated for updated camera {camera.name}")
            except Exception as e:
                logger.warning(f"Failed to auto-generate thumbnail for updated camera {camera.name}: {e}")
        
        # Handle AJAX POST requests
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'success': True,
                'message': f'Camera "{camera.name}" has been updated.'
            })
        
        messages.success(request, f'Camera "{camera.name}" has been updated.')
        return redirect('camera_settings')
    
    # Handle AJAX GET requests for modal editing
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # Get available AI modules for AJAX response
        aimodules = Aimodule.objects.filter(is_active=True).order_by('name')
        return JsonResponse({
            'id': camera.id,
            'name': camera.name,
            'rtsp_url': camera.rtsp_url,
            'username': camera.username or '',
            'camera_source': camera.camera_source,
            'is_active': camera.is_active,
            'created_at': camera.created_at.isoformat(),
            'roi_coordinates': camera.roi_coordinates,
            'thumbnail': camera.thumbnail.url if camera.thumbnail else None,
            'aimodule_id': camera.aimodule.id if camera.aimodule else None,
            'aimodule_name': camera.aimodule.name if camera.aimodule else None,
            'is_aimodule_active': camera.is_aimodule_active,
            'site_name': camera.site_name or '',
            'floor': camera.floor or '',
            'location': camera.location or '',
            'available_aimodules': [{'id': am.id, 'name': am.name, 'description': am.description} for am in aimodules]
        })
    
    # Get available AI modules
    aimodules = Aimodule.objects.filter(is_active=True).order_by('name')
    
    # Regular GET request returns template for standalone edit page
    context = {
        'camera': camera,
        'aimodules': aimodules,
        'page_title': 'Edit Camera',
        'page_subtitle': f'Configure {camera.name}'
    }
    
    return render(request, 'core/camera_edit.html', context)

@permission_required('core.delete_camera', raise_exception=True)
def camera_delete(request, camera_id):
    """Delete camera view"""
    if request.method == 'POST':
        try:
            camera = get_object_or_404(Camera, id=camera_id)
            camera_name = camera.name
            
            # Remove MediaMTX configuration
            try: 
                remove_result = remove_mediamtx_source(camera_name)
                if remove_result:
                    logger.info(f"MediaMTX configuration removed for {camera_name}")
            except Exception as e:
                logger.warning(f"Warning: Error removing MediaMTX configuration: {e}")
            
            # Delete camera from database
            camera.delete()
            
            # Handle AJAX requests
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'message': f'Camera "{camera_name}" has been deleted successfully.'
                })
            
            messages.success(request, f'Camera "{camera_name}" has been deleted successfully.')
            return redirect('camera_settings')
            
        except Camera.DoesNotExist:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': False,
                    'message': 'Camera not found.'
                })
            messages.error(request, 'Camera not found.')
            return redirect('camera_settings')
        except Exception as e:
            logger.error(f"Error deleting camera: {e}")
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': False,
                    'message': 'An error occurred while deleting the camera.'
                })
            messages.error(request, 'An error occurred while deleting the camera.')
            return redirect('camera_settings')
    
    # GET request not allowed
    return redirect('camera_settings')

@login_required(login_url='/login/')
def snapshots(request):
    """View snapshots and detections"""
    # Get filter parameters
    camera_id = request.GET.get('camera')
    date_from = request.GET.get('date_from')
    date_to = request.GET.get('date_to')
    
    # Build query
    detections = Detection.objects.select_related('camera').order_by('-timestamp')
    
    if camera_id:
        detections = detections.filter(camera_id=camera_id)
    
    if date_from:
        detections = detections.filter(timestamp__date__gte=date_from)
    
    if date_to:
        detections = detections.filter(timestamp__date__lte=date_to)
    
    # Pagination
    paginator = Paginator(detections, 12)  # Show 12 snapshots per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Get cameras for filter dropdown
    cameras = Camera.objects.all().order_by('name')
    
    context = {
        'page_obj': page_obj,
        'cameras': cameras,
        'selected_camera': camera_id,
        'date_from': date_from,
        'date_to': date_to,
        'page_title': 'Snapshots',
        'page_subtitle': 'Browse detection snapshots'
    }
    
    return render(request, 'core/snapshots.html', context)

@login_required(login_url='/login/')
def alerts(request):
    """View and manage alerts and detections"""
    # Get filter parameters
    severity = request.GET.get('severity')
    status = request.GET.get('status')
    camera_id = request.GET.get('camera')
    
    # Build alerts query
    alerts_query = Alert.objects.select_related('camera', 'detection').order_by('-created_at')
    
    if severity:
        alerts_query = alerts_query.filter(severity=severity)
    
    if status == 'acknowledged':
        alerts_query = alerts_query.filter(is_acknowledged=True)
    elif status == 'unacknowledged':
        alerts_query = alerts_query.filter(is_acknowledged=False)
    
    if camera_id:
        alerts_query = alerts_query.filter(camera_id=camera_id)
    
    # Build detections query
    detections_query = Detection.objects.select_related('camera').order_by('-timestamp')
    
    if camera_id:
        detections_query = detections_query.filter(camera_id=camera_id)
    
    # Pagination for alerts
    alerts_paginator = Paginator(alerts_query, 20)  # Show 20 alerts per page
    alerts_page_number = request.GET.get('alerts_page')
    alerts_page_obj = alerts_paginator.get_page(alerts_page_number)
    
    # Pagination for detections
    detections_paginator = Paginator(detections_query, 20)  # Show 20 detections per page
    detections_page_number = request.GET.get('detections_page')
    detections_page_obj = detections_paginator.get_page(detections_page_number)
    
    # Get summary statistics
    alert_stats = Alert.objects.aggregate(
        total=Count('id'),
        critical=Count('id', filter=Q(severity='critical')),
        high=Count('id', filter=Q(severity='high')),
        medium=Count('id', filter=Q(severity='medium')),
        low=Count('id', filter=Q(severity='low')),
        unacknowledged=Count('id', filter=Q(is_acknowledged=False)),
        acknowledged=Count('id', filter=Q(is_acknowledged=True))
    )
    
    # Get detection statistics
    today = timezone.now().date()
    detection_stats = Detection.objects.aggregate(
        total=Count('id'),
        today=Count('id', filter=Q(timestamp__date=today)),
        this_week=Count('id', filter=Q(timestamp__date__gte=today - timedelta(days=7))),
        this_month=Count('id', filter=Q(timestamp__date__gte=today - timedelta(days=30)))
    )
    
    # Get cameras for filter dropdown
    cameras = Camera.objects.all().order_by('name')
    
    context = {
        'alerts_page_obj': alerts_page_obj,
        'detections_page_obj': detections_page_obj,
        'alert_stats': alert_stats,
        'detection_stats': detection_stats,
        'cameras': cameras,
        'selected_severity': severity,
        'selected_status': status,
        'selected_camera': camera_id,
        'critical_count': alert_stats['critical'],
        'high_count': alert_stats['high'],
        'medium_count': alert_stats['medium'],
        'low_count': alert_stats['low'],
        'page_title': 'Security Dashboard',
        'page_subtitle': 'Monitor alerts and detections'
    }
    
    return render(request, 'core/alerts.html', context)

@login_required(login_url='/login/')
def alert_acknowledge(request, alert_id):
    """Acknowledge an alert"""
    if request.method == 'POST':
        alert = get_object_or_404(Alert, id=alert_id)
        alert.is_acknowledged = True
        alert.save()
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'status': 'success', 'message': 'Alert acknowledged'})
        else:
            messages.success(request, 'Alert has been acknowledged.')
    
    return redirect('alerts')

@login_required(login_url='/login/')
def detections(request):
    """View all detections"""
    # Get filter parameters
    camera_id = request.GET.get('camera')
    date_from = request.GET.get('date_from')
    date_to = request.GET.get('date_to')
    
    # Build query
    detections_query = Detection.objects.select_related('camera').order_by('-timestamp')
    
    if camera_id:
        detections_query = detections_query.filter(camera_id=camera_id)
    
    if date_from:
        detections_query = detections_query.filter(timestamp__date__gte=date_from)
    
    if date_to:
        detections_query = detections_query.filter(timestamp__date__lte=date_to)
    
    # Pagination
    paginator = Paginator(detections_query, 25)  # Show 25 detections per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Get cameras for filter dropdown
    cameras = Camera.objects.all().order_by('name')
    
    context = {
        'page_obj': page_obj,
        'cameras': cameras,
        'selected_camera': camera_id,
        'date_from': date_from,
        'date_to': date_to,
        'page_title': 'Detections',
        'page_subtitle': 'View detection history'
    }
    
    return render(request, 'core/detections.html', context)

@login_required(login_url='/login/')
def aicamera(request):
    """AI Camera management view"""
    # Get all cameras with their AI modules
    cameras = Camera.objects.select_related('aimodule').all().order_by('name')
    aimodules = Aimodule.objects.filter(is_active=True).order_by('name')
    
    # Calculate statistics
    total_cameras = cameras.count()
    ai_enabled_cameras = cameras.filter(aimodule__isnull=False).count()
    online_cameras = cameras.filter(is_active=True).count()
    offline_cameras = cameras.filter(is_active=False).count()
    
    # Get today's detection statistics
    today = timezone.now().date()
    today_detections = Detection.objects.filter(timestamp__date=today).count()
    
    # Get alert statistics
    alert_stats = Alert.objects.aggregate(
        total_alerts=Count('id', filter=Q(is_acknowledged=False)),
        critical_alerts=Count('id', filter=Q(is_acknowledged=False, severity='critical'))
    )
    
    context = {
        'cameras': cameras,
        'aimodules': aimodules,
        'total_cameras': total_cameras,
        'ai_enabled_cameras': ai_enabled_cameras,
        'online_cameras': online_cameras,
        'offline_cameras': offline_cameras,
        'today_detections': today_detections,
        'total_alerts': alert_stats['total_alerts'],
        'critical_alerts': alert_stats['critical_alerts'],
        'page_title': 'AI Camera',
        'page_subtitle': 'Cameras with AI Detection Modules'
    }
    
    return render(request, 'core/aicamera.html', context)

@login_required(login_url='/login/')
def aicamera_redirect(request):
    """Redirect old aicamera URL to new all_ai URL"""
    return redirect('all_ai')

@login_required(login_url='/login/')
def system_config(request):
    """System configuration view"""
    if request.method == 'POST':
        # Handle configuration updates
        config_data = {}
        
        # First pass: collect all config values
        for key, value in request.POST.items():
            if key.startswith('config_'):
                config_key = key.replace('config_', '')
                config_data[config_key] = {'value': value}
        
        # Second pass: collect categories
        for key, value in request.POST.items():
            if key.startswith('category_'):
                config_key = key.replace('category_', '')
                if config_key in config_data:
                    config_data[config_key]['category'] = value
        
        # Third pass: collect descriptions
        for key, value in request.POST.items():
            if key.startswith('description_'):
                config_key = key.replace('description_', '')
                if config_key in config_data:
                    config_data[config_key]['description'] = value
        
        # Now update or create configurations
        for config_key, data in config_data.items():
            config_obj, created = SystemConfig.objects.get_or_create(
                key=config_key,
                defaults={
                    'value': data['value'],
                    'category': data.get('category', 'general'),
                    'description': data.get('description', '')
                }
            )
            
            # Update all fields for existing objects
            config_obj.value = data['value']
            if 'category' in data:
                config_obj.category = data['category']
            if 'description' in data:
                config_obj.description = data['description']
            config_obj.save()
        
        messages.success(request, 'System configuration has been updated.')
        return redirect('system_config')
    
    # Get configuration grouped by category
    configs = SystemConfig.objects.all().order_by('category', 'key')
    
    # Group configs by category
    config_groups = {}
    for config in configs:
        if config.category not in config_groups:
            config_groups[config.category] = []
        config_groups[config.category].append(config)
    
    context = {
        'config_groups': config_groups,
        'page_title': 'System Configuration',
        'page_subtitle': 'Manage system settings'
    }
    
    return render(request, 'core/system_config.html', context)

@login_required(login_url='/login/')
def api_system_status(request):
    """API endpoint for system status (for AJAX updates)"""
    if request.method == 'GET':
        total_cameras = Camera.objects.count()
        active_cameras = Camera.objects.filter(is_active=True).count()
        unack_alerts = Alert.objects.filter(is_acknowledged=False).count()
        
        today = timezone.now().date()
        today_detections = Detection.objects.filter(timestamp__date=today).count()
        
        return JsonResponse({
            'total_cameras': total_cameras,
            'active_cameras': active_cameras,
            'unack_alerts': unack_alerts,
            'today_detections': today_detections,
            'system_load': 45,  # This would come from actual system monitoring
            'timestamp': timezone.now().isoformat()
        })
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@login_required(login_url='/login/')
@require_http_methods(["GET"])
def api_camera_details(request, camera_id):
    """API endpoint to get camera details"""
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        
        # Get recent detections for this camera
        recent_detections = Detection.objects.filter(
            camera=camera
        ).order_by('-timestamp')[:5]
        
        # Serialize detection data
        detections_data = []
        for detection in recent_detections:
            detections_data.append({
                'id': detection.id,
                'description': detection.description,
                'timestamp': detection.timestamp.isoformat(),
                'annotated_snapshot_path': detection.annotated_snapshot_path
            })
        
        camera_data = {
            'id': camera.id,
            'name': camera.name,
            'rtsp_url': camera.rtsp_url,
            'username': camera.username,
            'is_active': camera.is_active,
            'roi_coordinates': camera.roi_coordinates,
            'created_at': camera.created_at.isoformat(),
            'recent_detections': detections_data
        }
        
        return JsonResponse(camera_data)
        
    except Camera.DoesNotExist:
        return JsonResponse({'error': 'Camera not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@login_required(login_url='/login/')
@require_http_methods(["GET"])
def api_camera_stream_info(request, camera_id):
    """API endpoint to get camera streaming information"""
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        
        # Check if camera is active
        if not camera.is_active:
            return JsonResponse({
                'error': 'Camera is not active',
                'hls_available': False
            }, status=400)
        
        # Check real HLS stream availability
        hls_check = check_hls_stream_availability(camera.name)
        
        stream_info = {
            'camera_id': camera.id,
            'camera_name': camera.name,
            'hls_available': hls_check['available'],
            'hls_url': hls_check['url'] if hls_check['available'] else None,
            'hls_error': hls_check['error'] if not hls_check['available'] else None,
            'stream_quality': 'HD' if hls_check['available'] else 'Unavailable',
            'fps': 25,
            'resolution': '1280x720',
            'mediamtx_status': {
                'hls_status_code': hls_check['status_code'],
                'hls_content_type': hls_check['content_type']
            }
        }
        
        return JsonResponse(stream_info)
        
    except Camera.DoesNotExist:
        return JsonResponse({'error': 'Camera not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@permission_required('core.change_camera', raise_exception=True)
@csrf_exempt
@require_http_methods(["POST", "GET"])
def api_camera_test_connection(request, camera_id):
    """API endpoint to test camera connection with non-blocking operation"""
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        
        # Handle GET requests for task status polling
        if request.method == 'GET' and request.GET.get('task_id'):
            from .tasks import camera_task_manager
            task_id = request.GET.get('task_id')
            result = camera_task_manager.get_task_result(task_id)
            
            if result:
                if result['status'] == 'completed':
                    return JsonResponse(result['result'])
                elif result['status'] == 'failed':
                    return JsonResponse({
                        'success': False,
                        'message': result['error'],
                        'camera_id': camera.id
                    }, status=500)
                elif result['status'] == 'timeout':
                    return JsonResponse({
                        'success': False,
                        'message': 'Connection test timed out',
                        'camera_id': camera.id
                    }, status=408)
                else:
                    return JsonResponse({
                        'task_id': task_id,
                        'status': result['status'],
                        'message': 'Connection test in progress...'
                    })
            else:
                return JsonResponse({'error': 'Task not found'}, status=404)
        
        # Handle POST requests to start new connection test
        elif request.method == 'POST':
            from .tasks import camera_task_manager
            
            try:
                task_id = camera_task_manager.test_camera_connection_async(camera_id, timeout=15)
                
                return JsonResponse({
                    'task_id': task_id,
                    'status': 'pending',
                    'message': 'Connection test started...',
                    'camera_id': camera.id,
                    'check_url': f'/api/camera/{camera_id}/test-connection/?task_id={task_id}'
                })
                
            except Exception as e:
                return JsonResponse({
                    'success': False,
                    'message': f'Failed to start connection test: {str(e)}',
                    'camera_id': camera.id
                }, status=500)
        
        else:
            return JsonResponse({'error': 'Invalid request method or missing parameters'}, status=400)
            
    except Camera.DoesNotExist:
        return JsonResponse({'error': 'Camera not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@login_required(login_url='/login/')
@require_http_methods(["GET"])
def api_camera_health(request, camera_id):
    """API endpoint to check camera health status with non-blocking operation"""
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        
        if not camera.is_active:
            return JsonResponse({
                'camera_id': camera.id,
                'camera_name': camera.name,
                'status': 'inactive',
                'error': 'Camera is marked as inactive',
                'last_check': timezone.now().isoformat()
            })
        
        # Use async manager for non-blocking health check
        from .tasks import camera_task_manager
        
        # Check for cached status first
        cache_key = f"camera_health_{camera_id}"
        cached_status = cache.get(cache_key)
        
        if cached_status:
            return JsonResponse(cached_status)
        
        # Submit health check task
        try:
            task_id = camera_task_manager.get_camera_status_async(camera_id, timeout=10)
            result = camera_task_manager.get_task_result(task_id, wait=True, timeout=10)
            
            if result and result['status'] == 'completed':
                health_data = result['result']
                
                # Enhanced health response
                response_data = {
                    'camera_id': camera.id,
                    'camera_name': camera.name,
                    'status': 'healthy' if health_data['success'] else 'failed',
                    'quality': 'HD' if health_data['success'] else 'ERROR',
                    'fps': health_data.get('fps', 25),
                    'resolution': health_data.get('resolution', '1280x720'),
                    'latency': '< 200ms',
                    'error': None if health_data['success'] else health_data.get('message'),
                    'last_check': timezone.now().isoformat()
                }
                
                # Cache result for 30 seconds
                cache.set(cache_key, response_data, 30)
                return JsonResponse(response_data)
            else:
                return JsonResponse({
                    'camera_id': camera.id,
                    'camera_name': camera.name,
                    'status': 'timeout',
                    'error': 'Health check timed out',
                    'last_check': timezone.now().isoformat()
                }, status=408)
                
        except Exception as health_error:
            return JsonResponse({
                'camera_id': camera.id,
                'camera_name': camera.name,
                'status': 'error',
                'error': f'Health check failed: {str(health_error)}',
                'last_check': timezone.now().isoformat()
            }, status=500)
        
    except Camera.DoesNotExist:
        return JsonResponse({'error': 'Camera not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@login_required(login_url='/login/')
@csrf_exempt
@require_http_methods(["POST"])
def api_alerts_acknowledge_all(request):
    """API endpoint to acknowledge all unacknowledged alerts"""
    try:
        # Get all unacknowledged alerts
        unack_alerts = Alert.objects.filter(is_acknowledged=False)
        count = unack_alerts.count()
        
        # Acknowledge all alerts
        unack_alerts.update(is_acknowledged=True)
        
        return JsonResponse({
            'success': True,
            'message': f'Successfully acknowledged {count} alerts',
            'count': count,
            'timestamp': timezone.now().isoformat()
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# Old functions removed - now using optimized rtsp_connect module

@login_required(login_url='/login/')
@require_http_methods(["GET"])
def api_camera_stream_hls(request, camera_id):
    """Mock HLS streaming endpoint"""
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        
        if not camera.is_active:
            return JsonResponse({'error': 'Camera is not active'}, status=400)
        
        # For now, return a mock HLS playlist
        # In production, this would generate or proxy actual HLS content
        mock_playlist = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:10
#EXT-X-MEDIA-SEQUENCE:0
#EXT-X-PLAYLIST-TYPE:EVENT
#EXTINF:10.0,
segment0.ts
#EXTINF:10.0,
segment1.ts
#EXTINF:10.0,
segment2.ts
#EXT-X-ENDLIST
"""
        
        from django.http import HttpResponse
        return HttpResponse(mock_playlist, content_type='application/vnd.apple.mpegurl')
        
    except Camera.DoesNotExist:
        return JsonResponse({'error': 'Camera not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)



@login_required(login_url='/login/')
@require_http_methods(["GET"])
def camera_live_stream(request, camera_id):
    """Direct live camera stream view - redirects to HLS stream"""
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        
        if not camera.is_active:
            from django.http import HttpResponse
            return HttpResponse("Camera is not active", status=400)
        
        # Check if HLS stream is available
        hls_check = check_hls_stream_availability(camera.name)
        
        if hls_check['available']:
            # Redirect to MediaMTX HLS stream
            from django.http import HttpResponseRedirect
            return HttpResponseRedirect(hls_check['url'])
        else:
            # Return error if HLS stream not available
            from django.http import HttpResponse
            error_msg = hls_check['error'] or 'HLS stream not available'
            logger.warning(f"HLS stream unavailable for camera {camera.name}: {error_msg}")
            return HttpResponse(f"Stream unavailable: {error_msg}", status=404)
        
    except Camera.DoesNotExist:
        from django.http import HttpResponse
        return HttpResponse("Camera not found", status=404)
    except Exception as e:
        from django.http import HttpResponse
        logger.error(f"Stream error for camera {camera_id}: {e}")
        return HttpResponse(f"Stream error: {str(e)}", status=500)

def permission_denied_view(request, exception):
    """
    Custom 403 permission denied view
    """
    return render(request, 'core/403.html', status=403)

@login_required(login_url='/login/')
@require_http_methods(["GET"])
def api_camera_hls_check(request, camera_id):
    """API endpoint to check HLS stream availability for a specific camera"""
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        
        # Check HLS stream availability
        hls_check = check_hls_stream_availability(camera.name)
        
        response_data = {
            'camera_id': camera.id,
            'camera_name': camera.name,
            'hls_check': hls_check,
            'expected_url': hls_check['url'],
            'timestamp': timezone.now().isoformat()
        }
        
        if hls_check['available']:
            return JsonResponse(response_data)
        else:
            return JsonResponse(response_data, status=404)
        
    except Camera.DoesNotExist:
        return JsonResponse({'error': 'Camera not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@permission_required('core.change_camera', raise_exception=True)
@require_http_methods(["POST"])
def api_camera_capture_frame(request, camera_id):
    """API endpoint to capture a single frame from camera"""
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        
        # Import the capture function
        from .rtsp_connect import capture_single_frame
        
        # Capture frame using existing functionality
        base64_frame = capture_single_frame(camera_id)
        
        if base64_frame:
            return JsonResponse({
                'success': True,
                'camera_id': camera.id,
                'camera_name': camera.name,
                'frame_data': base64_frame,
                'timestamp': timezone.now().isoformat(),
                'format': 'base64_jpeg'
            })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Failed to capture frame from camera',
                'camera_id': camera.id
            }, status=500)
            
    except Camera.DoesNotExist:
        return JsonResponse({'error': 'Camera not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def generate_and_save_thumbnail(camera):
    """
    Generate a thumbnail for a camera and save it to the database
    Returns True if successful, False otherwise
    """
    try:
        from .rtsp_connect import capture_single_frame
        from django.core.files.base import ContentFile
        import base64
        import uuid
        
        logger.info(f"Generating thumbnail for camera {camera.id}: {camera.name}")
        
        # Capture frame using existing functionality
        base64_frame = capture_single_frame(camera.id)
        
        if base64_frame:
            # Convert base64 to bytes
            image_data = base64.b64decode(base64_frame)
            
            # Create a Django file object
            filename = f"camera_{camera.id}_{uuid.uuid4().hex[:8]}.jpg"
            thumbnail_file = ContentFile(image_data, name=filename)
            
            # Save to camera's thumbnail field
            camera.thumbnail.save(filename, thumbnail_file, save=True)
            
            logger.info(f"Successfully generated thumbnail for camera {camera.id}: {camera.name}")
            return True
        else:
            logger.warning(f"Failed to capture frame for camera {camera.id}: {camera.name}")
            return False
            
    except Exception as e:
        logger.error(f"Error generating thumbnail for camera {camera.id}: {str(e)}")
        return False


@permission_required('core.change_camera', raise_exception=True)
@require_http_methods(["POST"])
def api_generate_thumbnail(request, camera_id):
    """API endpoint to generate and save a thumbnail for a camera"""
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        
        success = generate_and_save_thumbnail(camera)
        
        if success:
            return JsonResponse({
                'success': True,
                'camera_id': camera.id,
                'camera_name': camera.name,
                'thumbnail_url': camera.thumbnail.url if camera.thumbnail else None,
                'message': 'Thumbnail generated and saved successfully'
            })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Failed to generate thumbnail',
                'camera_id': camera.id
            }, status=500)
            
    except Camera.DoesNotExist:
        return JsonResponse({'error': 'Camera not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required(login_url='/login/')
@require_http_methods(["GET"])
def detection_security_intrusion(request, camera_id):
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        camera_name = camera.name
        camera_source = camera.camera_source
        mediamtx = SystemConfig.objects.get(key='mediamtx_IP').value
        
        # Check if camera is active
        if not camera.is_active:
            logger.warning(f"Attempted to access inactive camera {camera_name}")
            return HttpResponse("Camera is not active", status=400)
        
        # Use HLS URL for video source
        # Use HLS URL for video source
        url = camera.rtsp_url
        if url[0:5] == 'local':
            HLS_URL = url
        else:
            if camera_source == 'RTSP Stream':
                #HLS_URL = f"{mediamtx}:8888/{camera_name}/index.m3u8"
                HLS_URL = f"rtsp://{camera.username}:{camera.password}@{camera.rtsp_url.split('rtsp://')[1]}"
            else:
                full_rtsp_url = f"rtsp://{camera.username}:{camera.password}@{camera.rtsp_url.split('rtsp://')[1]}"
                HLS_URL = full_rtsp_url

        logger.info(f"Starting security intrusion detection for camera {camera_name}")
        
        # Create a streaming generator wrapper that forces immediate transmission
        def stream_wrapper():
            try:
                import sys
                for chunk in sec.start_detection_stream(camera, HLS_URL):
                    yield chunk
                    # Force flush in case server is buffering
                    sys.stdout.flush()
            except Exception as e:
                logger.error(f"Stream error: {e}")
                # Yield a simple error frame
                error_frame = b'--frame\r\nContent-Type: image/jpeg\r\n\r\nERROR\r\n'
                yield error_frame
        
        # Start detection stream with automatic thread management
        response = StreamingHttpResponse(
            stream_wrapper(),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
        # Headers to prevent any buffering
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        response['X-Accel-Buffering'] = 'no'
        response['Access-Control-Allow-Origin'] = '*'
        response.streaming = True  # Ensure Django treats this as streaming
        return response
            
    except Camera.DoesNotExist:
        logger.error(f"Camera with ID {camera_id} not found")
        return HttpResponse("Camera not found", status=404)
    except Exception as e:
        logger.error(f"Error in detection_security_intrusion for camera {camera_id}: {e}")
        return HttpResponse(f"System error: {str(e)}", status=500)


@login_required(login_url='/login/')
@require_http_methods(["GET"])
def detection_ppe(request, camera_id):
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        camera_name = camera.name
        camera_source = camera.camera_source
        mediamtx = SystemConfig.objects.get(key='mediamtx_IP').value
       
        
        # Check if camera is active
        if not camera.is_active:
            logger.warning(f"Attempted to access inactive camera {camera_name}")
            return HttpResponse("Camera is not active", status=400)
        
        # Use HLS URL for video source
        url = camera.rtsp_url
        if url[0:5] == 'local':
            HLS_URL = url
        else:
            if camera_source == 'RTSP Stream':
                #HLS_URL = f"{mediamtx}:8888/{camera_name}/index.m3u8"
                HLS_URL = f"rtsp://{camera.username}:{camera.password}@{camera.rtsp_url.split('rtsp://')[1]}"
            else:
                full_rtsp_url = f"rtsp://{camera.username}:{camera.password}@{camera.rtsp_url.split('rtsp://')[1]}"
                HLS_URL = full_rtsp_url

        logger.info(f"Starting PPE detection for camera {camera_name}")
        
        # Create a streaming generator wrapper for ASGI compatibility
        def stream_wrapper():
            try:
                for chunk in ppe.start_detection_stream(camera, HLS_URL):
                    yield chunk
            except Exception as e:
                logger.error(f"PPE Stream error: {e}")
                error_frame = b'--frame\r\nContent-Type: image/jpeg\r\n\r\nERROR\r\n'
                yield error_frame
        
        # Start detection stream with automatic thread management
        response = StreamingHttpResponse(
            stream_wrapper(),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['X-Accel-Buffering'] = 'no'
        return response
            
    except Camera.DoesNotExist:
        logger.error(f"Camera with ID {camera_id} not found")
        return HttpResponse("Camera not found", status=404)
    except Exception as e:
        logger.error(f"Error in detection_security_intrusion for camera {camera_id}: {e}")
        return HttpResponse(f"System error: {str(e)}", status=500)

@login_required(login_url='/login/')
@require_http_methods(["GET"])
def detection_object(request, camera_id):
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        camera_name = camera.name
        mediamtx = SystemConfig.objects.get(key='mediamtx_IP').value
        camera_source = camera.camera_source
        
        # Check if camera is active
        if not camera.is_active:
            logger.warning(f"Attempted to access inactive camera {camera_name}")
            return HttpResponse("Camera is not active", status=400)
        
        # Use HLS URL for video source
        
        url = camera.rtsp_url
        if camera_source == 'local' or camera_source == 'http' or camera_source == 'youtube' or camera_source == 'video_file':
            HLS_URL = url
        else:
            HLS_URL = f"{mediamtx}:8888/{camera_name}/index.m3u8"

        logger.info(f"Starting object detection for camera {camera_name}")
        
        # Create a streaming generator wrapper for ASGI compatibility
        def stream_wrapper():
            try:
                for chunk in obj.start_detection_stream(camera, HLS_URL):
                    yield chunk
            except Exception as e:
                logger.error(f"Object detection stream error: {e}")
                error_frame = b'--frame\r\nContent-Type: image/jpeg\r\n\r\nERROR\r\n'
                yield error_frame
        
        # Start detection stream with automatic thread management
        response = StreamingHttpResponse(
            stream_wrapper(),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['X-Accel-Buffering'] = 'no'
        return response
            
    except Camera.DoesNotExist:
        logger.error(f"Camera with ID {camera_id} not found")
        return HttpResponse("Camera not found", status=404)
    except Exception as e:
        logger.error(f"Error in detection_security_intrusion for camera {camera_id}: {e}")
        return HttpResponse(f"System error: {str(e)}", status=500)

@login_required(login_url='/login/')
@require_http_methods(["GET"])
def detection_object_video_source(request, camera_id):
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        camera_name = camera.name
        mediamtx = SystemConfig.objects.get(key='mediamtx_IP').value
        camera_source = camera.camera_source
        
        # Check if camera is active
        if not camera.is_active:
            logger.warning(f"Attempted to access inactive camera {camera_name}")
            return HttpResponse("Camera is not active", status=400)
        
        # Use HLS URL for video source
        
        url = camera.rtsp_url
        if camera_source == 'local' or camera_source == 'http' or camera_source == 'youtube' or camera_source == 'video_file':
            HLS_URL = url
        else:
            HLS_URL = f"{mediamtx}:8888/{camera_name}/index.m3u8"

        logger.info(f"Starting object detection for camera {camera_name}")
        
        # Create a streaming generator wrapper for ASGI compatibility
        def stream_wrapper():
            try:
                for chunk in obj_det_vid_source.start_detection_stream(camera, HLS_URL):
                    yield chunk
            except Exception as e:
                logger.error(f"Object detection stream error: {e}")
                error_frame = b'--frame\r\nContent-Type: image/jpeg\r\n\r\nERROR\r\n'
                yield error_frame
        
        # Start detection stream with automatic thread management
        response = StreamingHttpResponse(
            stream_wrapper(),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['X-Accel-Buffering'] = 'no'
        return response
            
    except Camera.DoesNotExist:
        logger.error(f"Camera with ID {camera_id} not found")
        return HttpResponse("Camera not found", status=404)
    except Exception as e:
        logger.error(f"Error in detection_security_intrusion for camera {camera_id}: {e}")
        return HttpResponse(f"System error: {str(e)}", status=500)


# Object Detection with CPU/GPU
@login_required(login_url='/login/')
@require_http_methods(["GET"])
def det_obj_cpu(request, camera_id):
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        camera_name = camera.name
        mediamtx = SystemConfig.objects.get(key='mediamtx_IP').value
        
        # Check if camera is active
        if not camera.is_active:
            logger.warning(f"Attempted to access inactive camera {camera_name}")
            return HttpResponse("Camera is not active", status=400)
        
        # Use HLS URL for video source
        url = camera.rtsp_url
        if url[0:5] == 'local':
            HLS_URL = url
        else:
            HLS_URL = f"{mediamtx}:8888/{camera_name}/index.m3u8"
        
        logger.info(f"Starting object detection for camera {camera_name}")
        
        # Start detection stream with automatic thread management
        return StreamingHttpResponse(
            detObjCpu.start_detection_stream(camera, HLS_URL),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
            
    except Camera.DoesNotExist:
        logger.error(f"Camera with ID {camera_id} not found")
        return HttpResponse("Camera not found", status=404)
    except Exception as e:
        logger.error(f"Error in detection_security_intrusion for camera {camera_id}: {e}")
        return HttpResponse(f"System error: {str(e)}", status=500)


# PPE Detection with CPU
@login_required(login_url='/login/')
@require_http_methods(["GET"])
def ppe_det_cpu(request, camera_id):
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        camera_name = camera.name
        mediamtx = SystemConfig.objects.get(key='mediamtx_IP').value
        
        # Check if camera is active
        if not camera.is_active:
            logger.warning(f"Attempted to access inactive camera {camera_name}")
            return HttpResponse("Camera is not active", status=400)
        
        # Use HLS URL for video source
        url = camera.rtsp_url
        if url[0:5] == 'local':
            HLS_URL = url
        else:
            HLS_URL = f"{mediamtx}:8888/{camera_name}/index.m3u8"
        
        logger.info(f"Starting object detection for camera {camera_name}")
        
        # Start detection stream with automatic thread management
        return StreamingHttpResponse(
            ppe_cpu.start_detection_stream(camera, HLS_URL),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
            
    except Camera.DoesNotExist:
        logger.error(f"Camera with ID {camera_id} not found")
        return HttpResponse("Camera not found", status=404)
    except Exception as e:
        logger.error(f"Error in detection_security_intrusion for camera {camera_id}: {e}")
        return HttpResponse(f"System error: {str(e)}", status=500)


# PPE Detection with CPU
@login_required(login_url='/login/')
@require_http_methods(["GET"])
def obj_det_cpu(request, camera_id):
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        camera_name = camera.name
        mediamtx = SystemConfig.objects.get(key='mediamtx_IP').value
        
        # Check if camera is active
        if not camera.is_active:
            logger.warning(f"Attempted to access inactive camera {camera_name}")
            return HttpResponse("Camera is not active", status=400)
        
        # Use HLS URL for video source
        url = camera.rtsp_url
        if url[0:5] == 'local':
            HLS_URL = url
        else:
            HLS_URL = f"{mediamtx}:8888/{camera_name}/index.m3u8"
        
        logger.info(f"Starting object detection for camera {camera_name}")
        
        # Start detection stream with automatic thread management
        return StreamingHttpResponse(
            obj_cpu.start_detection_stream(camera, HLS_URL),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
            
    except Camera.DoesNotExist:
        logger.error(f"Camera with ID {camera_id} not found")
        return HttpResponse("Camera not found", status=404)
    except Exception as e:
        logger.error(f"Error in detection_security_intrusion for camera {camera_id}: {e}")
        return HttpResponse(f"System error: {str(e)}", status=500)


# PPE Detection with onnx
@login_required(login_url='/login/')
@require_http_methods(["GET"])
def obj_det_onnx(request, camera_id):
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        camera_name = camera.name
        mediamtx = SystemConfig.objects.get(key='mediamtx_IP').value
        
        # Check if camera is active
        if not camera.is_active:
            logger.warning(f"Attempted to access inactive camera {camera_name}")
            return HttpResponse("Camera is not active", status=400)
        
        # Use HLS URL for video source
        url = camera.rtsp_url
        if url[0:5] == 'local':
            HLS_URL = url
        else:
            HLS_URL = f"{mediamtx}:8888/{camera_name}/index.m3u8"
        
        logger.info(f"Starting object detection for camera {camera_name}")
        
        # Start detection stream with automatic thread management
        return StreamingHttpResponse(
            obj_onnx.start_detection_stream(camera, HLS_URL),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
            
    except Camera.DoesNotExist:
        logger.error(f"Camera with ID {camera_id} not found")
        return HttpResponse("Camera not found", status=404)
    except Exception as e:
        logger.error(f"Error in detection_security_intrusion for camera {camera_id}: {e}")
        return HttpResponse(f"System error: {str(e)}", status=500)

# Object Detection with yoloe
@login_required(login_url='/login/')
@require_http_methods(["GET"])
def obj_det_yoloe(request, camera_id):
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        camera_name = camera.name
        mediamtx = SystemConfig.objects.get(key='mediamtx_IP').value
        
        # Check if camera is active
        if not camera.is_active:
            logger.warning(f"Attempted to access inactive camera {camera_name}")
            return HttpResponse("Camera is not active", status=400)
        
        # Use HLS URL for video source
        url = camera.rtsp_url
        if url[0:5] == 'local':
            HLS_URL = url
        else:
            HLS_URL = f"{mediamtx}:8888/{camera_name}/index.m3u8"
        
        logger.info(f"Starting object detection for camera {camera_name}")
        
        # Start detection stream with automatic thread management
        return StreamingHttpResponse(
            obj_yoloe.start_detection_stream(camera, HLS_URL),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
            
    except Camera.DoesNotExist:
        logger.error(f"Camera with ID {camera_id} not found")
        return HttpResponse("Camera not found", status=404)
    except Exception as e:
        logger.error(f"Error in detection_security_intrusion for camera {camera_id}: {e}")
        return HttpResponse(f"System error: {str(e)}", status=500)