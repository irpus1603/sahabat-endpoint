from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.decorators import login_required
from core.models import Camera, Detection, intrusionChecks, Alert
from django.shortcuts import redirect
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods
from django.http import StreamingHttpResponse
from core.models import SystemConfig
from . import security_intrusion as intr
from django.utils import timezone
from django.db.models import Count, Q
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime, timedelta
from django.conf import settings
import pytz
import json

import logging

import intrusion

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@login_required(login_url='/login/')
def list_intrusion_camera(request):
    """View to display list of cameras with PPE detection capabilities"""
    try:
        # Get all active cameras
        cameras = Camera.objects.filter(is_active=True)
        
        # Calculate statistics
        total_cameras = cameras.count()
        intrusion_enabled_cameras = cameras.filter(is_active=True).count()  
        online_cameras = cameras.filter(is_active=True).count()
                
        # Add mock compliance rates and violation counts for display
        
        context = {
            'cameras': cameras,
            'intrusion_enabled_cameras': intrusion_enabled_cameras,
            'online_cameras': online_cameras,
            'total_cameras': total_cameras,
        }
        
        return render(request, 'intrusion/list_intrusion_camera.html', context)
        
    except Exception as e:
        logger.error(f"Error in list_intrusion_camera: {e}")
        return HttpResponse(f"System error: {str(e)}", status=500)


@login_required(login_url='/login/')
@require_http_methods(["GET"])
def intrusion_detection(request, camera_id):
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        camera_name = camera.name
        camera_source = camera.camera_source
        mediamtx = SystemConfig.objects.get(key='mediamtx_IP').value
       
        
        # Check if camera is active
        if not camera.is_active:
            logger.warning(f"Attempted to access inactive camera {camera_name}")
            return HttpResponse("Camera is not active", status=400)
        
        hls_url = None
       
        
        # Check if camera is active
        if not camera.is_active:
            logger.warning(f"Attempted to access inactive camera {camera_name}")
            return HttpResponse("Camera is not active", status=400)
        
        # Use HLS URL for video source
        url = camera.rtsp_url
        if camera_source == 'video_file' and url.startswith('/media/'):
            hls_url = url[1:]  # Remove leading slash for local file paths
        elif camera_source == 'local':
            url = url.split('local:')[1]  # Extract local path
        elif camera_source == 'rtsp':
            hls_url = f"rtsp://{camera.username}:{camera.password}@{camera.rtsp_url.split('rtsp://')[1]}"         
        else:
            hls_url = url
        logger.info(f"Starting people counting detection for camera {camera_name}")

        
        
        # Create a streaming generator wrapper for ASGI compatibility
        def stream_wrapper():
            try:
                for chunk in intr.start_detection_stream(camera, hls_url):
                    yield chunk
            except Exception as e:
                logger.error(f"Intrusion Stream error: {e}")
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
def intrusion_detection_page(request, camera_id):
    """View to display PPE detection page with template"""
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        
        # Check if camera is active
        if not camera.is_active:
            logger.warning(f"Attempted to access inactive camera {camera.name}")
            return HttpResponse("Camera is not active", status=400)
        
        # Get intrusion data for the last 24 hours
        intrusion_data = get_intrusion_chart_data(camera_id)
        
        context = {
            'camera': camera,
            'current_time': timezone.now(),
            'intrusion_data': intrusion_data,
            'intrusion_data_json': json.dumps(intrusion_data),
        }

        return render(request, 'intrusion/detection_intrusion.html', context)

    except Camera.DoesNotExist:
        logger.error(f"Camera with ID {camera_id} not found")
        return HttpResponse("Camera not found", status=404)
    except Exception as e:
        logger.error(f"Error in intrusion_detection_page for camera {camera_id}: {e}")
        return HttpResponse(f"System error: {str(e)}", status=500)


def get_intrusion_chart_data(camera_id):
    """Get intrusion detection data for the last 6 hours with 30-minute intervals using Detection table only"""
    try:
        # Get timezone from settings
        local_tz = pytz.timezone(settings.TIME_ZONE)
        
        # Get current time in local timezone
        now_utc = timezone.now()
        now_local = now_utc.astimezone(local_tz)
        
        # Set time range (last 6 hours from current local time) - fix: include current hour
        end_time = now_local.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)  # Include current hour
        start_time = end_time - timedelta(hours=6)
        
        # Convert back to UTC for database queries (Django stores in UTC)
        start_time_utc = start_time.astimezone(pytz.UTC)
        end_time_utc = end_time.astimezone(pytz.UTC)
        
        logger.info(f"Getting intrusion data for camera {camera_id}")
        logger.info(f"Local time range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"UTC time range for query: {start_time_utc.strftime('%Y-%m-%d %H:%M')} to {end_time_utc.strftime('%Y-%m-%d %H:%M')}")
        
        # Debug: Check if there are any intrusion detections at all for this camera
        all_intrusions_for_camera = Detection.objects.filter(
            camera_id=camera_id,
            detection_type='intrusion'
        ).count()
        logger.info(f"Total intrusion detections for camera {camera_id}: {all_intrusions_for_camera}")
        
        # Debug: Check all detection types for this camera
        all_detections_for_camera = Detection.objects.filter(
            camera_id=camera_id
        ).values('detection_type').distinct()
        logger.info(f"All detection types for camera {camera_id}: {list(all_detections_for_camera)}")
        
        # Generate 30-minute interval labels and time slots in local time
        labels = []
        time_slots_local = []
        current_time = start_time
        
        while current_time < end_time:  # Changed <= to < to avoid extra slot
            labels.append(current_time.strftime('%H:%M'))
            time_slots_local.append(current_time)
            current_time += timedelta(minutes=30)
        
        # Convert local time slots to UTC for database queries
        time_slots_utc = []
        for slot in time_slots_local:
            time_slots_utc.append(slot.astimezone(pytz.UTC))
        
        # Get intrusion data for each time slot using Detection table only
        data = []
        total_intrusions = 0
        
        for i in range(len(time_slots_utc) - 1):
            slot_start_utc = time_slots_utc[i]
            slot_end_utc = time_slots_utc[i + 1]
            slot_start_local = time_slots_local[i]
            slot_end_local = time_slots_local[i + 1]
            
            # Count intrusions from Detection table directly
            query_filter = {
                'detection_type': 'intrusion',  # Filter for intrusion detection type
                'timestamp__gte': slot_start_utc,
                'timestamp__lt': slot_end_utc
            }
            
            # Add camera filter only if camera_id is provided
            if camera_id:
                query_filter['camera_id'] = camera_id
                
            intrusion_count = Detection.objects.filter(**query_filter).count()
            
            data.append(intrusion_count)
            total_intrusions += intrusion_count
            
            logger.debug(f"Local time slot {slot_start_local.strftime('%H:%M')}-{slot_end_local.strftime('%H:%M')}: {intrusion_count} intrusions")
        
        logger.info(f"Total intrusions found in Detection table: {total_intrusions}")
        
        # If no real data exists, check if we should show sample data
        if total_intrusions == 0:
            logger.warning(f"No intrusion data found in Detection table for camera {camera_id}")
            # Check if this is a demo/development environment
            try:
                demo_mode_value = SystemConfig.get_value('demo_mode', 'false')
                demo_mode = demo_mode_value and demo_mode_value.lower() == 'true'
                if demo_mode:
                    # Generate sample data for demonstration
                    import random
                    data = [random.randint(0, 3) for _ in data]
                    logger.info("Using demo sample data for intrusion chart")
                else:
                    # Return all zeros to show empty chart - more accurate for production
                    data = [0] * len(data)
            except:
                # Return all zeros to show empty chart - this is more accurate than fake data
                data = [0] * len(data)
        
        return {
            'labels': labels,
            'data': data
        }
        
    except Exception as e:
        logger.error(f"Error getting intrusion chart data from Detection table: {e}")
        # Return current time-based fallback data
        try:
            local_tz = pytz.timezone(settings.TIME_ZONE)
            now_local = timezone.now().astimezone(local_tz)
            end_time = now_local.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            start_time = end_time - timedelta(hours=6)
            
            labels = []
            current_time = start_time
            while current_time < end_time:
                labels.append(current_time.strftime('%H:%M'))
                current_time += timedelta(minutes=30)
        except:
            # Ultimate fallback with static labels
            labels = ['08:00', '08:30', '09:00', '09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00', '13:30']
        
        return {
            'labels': labels,
            'data': [0, 1, 0, 2, 1, 0, 1, 3, 0, 1, 0, 2]
        }


@login_required(login_url='/login/')
@csrf_exempt
def api_intrusion_violations(request, camera_id):
    """API endpoint for intrusion violations chart data"""
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        data = get_intrusion_chart_data(camera_id)
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Error in api_intrusion_violations: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@login_required(login_url='/login/')
@csrf_exempt  
def api_intrusion_summary(request, camera_id):
    """API endpoint for intrusion summary statistics"""
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        
        # Get timezone from settings
        local_tz = pytz.timezone(settings.TIME_ZONE)
        now_utc = timezone.now()
        now_local = now_utc.astimezone(local_tz)
        
        # Calculate time boundaries in UTC for database queries
        day_1_utc = (now_local - timedelta(days=1)).astimezone(pytz.UTC)
        day_7_utc = (now_local - timedelta(days=7)).astimezone(pytz.UTC)
        day_30_utc = (now_local - timedelta(days=30)).astimezone(pytz.UTC)
        
        logger.info(f"Local current time: {now_local.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate intrusion counts using Detection table with local timezone
        intrusions_1day = Detection.objects.filter(
            camera_id=camera_id,
            detection_type='intrusion',
            timestamp__gte=day_1_utc
        ).count()
        
        intrusions_7days = Detection.objects.filter(
            camera_id=camera_id, 
            detection_type='intrusion',
            timestamp__gte=day_7_utc
        ).count()
        
        intrusions_30days = Detection.objects.filter(
            camera_id=camera_id,
            detection_type='intrusion',
            timestamp__gte=day_30_utc
        ).count()
        
        # If no real data, check if we should provide sample counts for demonstration
        if intrusions_1day == 0 and intrusions_7days == 0 and intrusions_30days == 0:
            logger.warning(f"No intrusion summary data found for camera {camera_id}")
            try:
                demo_mode_value = SystemConfig.get_value('demo_mode', 'false')
                demo_mode = demo_mode_value and demo_mode_value.lower() == 'true'
                if demo_mode:
                    # Generate sample data for demonstration
                    import random
                    intrusions_1day = random.randint(0, 5)
                    intrusions_7days = intrusions_1day + random.randint(5, 15)
                    intrusions_30days = intrusions_7days + random.randint(10, 25)
                    logger.info("Using demo sample data for intrusion summary")
            except:
                # Keep zeros if not in demo mode or if there's an error
                pass
        
        data = {
            'intrusions_1day': intrusions_1day,
            'intrusions_7days': intrusions_7days,
            'intrusions_30days': intrusions_30days,
            'last_updated': now_local.strftime('%H:%M:%S')  # Local time for display
        }
        
        return JsonResponse(data)
        
    except Exception as e:
        logger.error(f"Error in api_intrusion_summary: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@login_required(login_url='/login/')
@csrf_exempt
def api_intrusion_severity_breakdown(request, camera_id):
    """API endpoint for intrusion severity breakdown"""
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        now = timezone.now()
        week_ago = now - timedelta(days=7)
        
        # Get severity data directly from intrusionChecks with Detection timestamp
        severity_data = intrusionChecks.objects.filter(
            camera_id=camera_id,
            intrusion=True,
            Detectionid__timestamp__gte=week_ago
        ).values('severity').annotate(count=Count('id')).order_by('severity')
        
        labels = []
        data = []
        
        # Define severity levels
        default_severities = ['low', 'medium', 'high', 'critical']
        severity_counts = {item['severity'] or 'unknown': item['count'] for item in severity_data}
        
        for severity in default_severities:
            labels.append(severity.title())
            data.append(severity_counts.get(severity, 0))
        
        # Add unknown severity if exists
        if 'unknown' in severity_counts:
            labels.append('Unknown')
            data.append(severity_counts['unknown'])
        
        # If no data, provide sample data for demo mode
        if all(count == 0 for count in data):
            try:
                demo_mode_value = SystemConfig.get_value('demo_mode', 'false')
                demo_mode = demo_mode_value and demo_mode_value.lower() == 'true'
                if demo_mode:
                    import random
                    data = [random.randint(0, 5) for _ in labels]
                    logger.info("Using demo sample data for severity breakdown")
            except:
                pass
        
        return JsonResponse({
            'labels': labels,
            'data': data
        })
        
    except Exception as e:
        logger.error(f"Error in api_intrusion_severity_breakdown: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@login_required(login_url='/login/')
@csrf_exempt
def api_intrusion_type_breakdown(request, camera_id):
    """API endpoint for intrusion type breakdown"""
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        now = timezone.now()
        week_ago = now - timedelta(days=7)
        
        # Get type data directly from intrusionChecks with Detection timestamp
        type_data = intrusionChecks.objects.filter(
            camera_id=camera_id,
            intrusion=True,
            Detectionid__timestamp__gte=week_ago
        ).values('intrusion_type').annotate(count=Count('id')).order_by('-count')
        
        labels = []
        data = []
        
        for item in type_data:
            intrusion_type = item['intrusion_type'] or 'Unknown'
            labels.append(intrusion_type.title())
            data.append(item['count'])
        
        # If no data, handle based on demo mode
        if not labels:
            try:
                demo_mode_value = SystemConfig.get_value('demo_mode', 'false')
                demo_mode = demo_mode_value and demo_mode_value.lower() == 'true'
                if demo_mode:
                    import random
                    labels = ['Unauthorized Entry', 'Restricted Area', 'After Hours', 'Unknown']
                    data = [random.randint(0, 10) for _ in labels]
                    logger.info("Using demo sample data for type breakdown")
                else:
                    labels = ['No Data']
                    data = [0]
            except:
                labels = ['No Data']
                data = [0]
        
        return JsonResponse({
            'labels': labels,
            'data': data
        })
        
    except Exception as e:
        logger.error(f"Error in api_intrusion_type_breakdown: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@login_required(login_url='/login/')
def intrusion_report(request):
    """Comprehensive security intrusion report"""
    try:
        # Get timezone from settings
        local_tz = pytz.timezone(settings.TIME_ZONE)
        now_utc = timezone.now()
        now_local = now_utc.astimezone(local_tz)
        
        # Default to last 7 days, but allow filtering
        date_range = request.GET.get('date_range', 'last7days')
        camera_filter = request.GET.get('camera_id', '')
        severity_filter = request.GET.get('severity', '')
        
        # Debug logging
        logger.info(f"Report filters - date_range: {date_range}, camera_filter: {camera_filter}, severity_filter: {severity_filter}")
        
        # Calculate time range based on filter
        if date_range == 'today':
            start_time = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = now_local.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif date_range == 'yesterday':
            yesterday = now_local - timedelta(days=1)
            start_time = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif date_range == 'last30days':
            start_time = now_local - timedelta(days=30)
            end_time = now_local
        elif date_range == 'custom':
            # Handle custom date range
            start_date_str = request.GET.get('start_date', '')
            end_date_str = request.GET.get('end_date', '')
            
            try:
                if start_date_str and end_date_str:
                    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
                    
                    start_time = local_tz.localize(datetime.combine(start_date, datetime.min.time()))
                    end_time = local_tz.localize(datetime.combine(end_date, datetime.max.time()))
                else:
                    # Fallback to last 7 days if custom dates are invalid
                    start_time = now_local - timedelta(days=7)
                    end_time = now_local
            except ValueError:
                # Fallback to last 7 days if custom dates are invalid
                start_time = now_local - timedelta(days=7)
                end_time = now_local
        else:  # last7days (default)
            start_time = now_local - timedelta(days=7)
            end_time = now_local
        
        # Convert to UTC for database queries
        start_time_utc = start_time.astimezone(pytz.UTC)
        end_time_utc = end_time.astimezone(pytz.UTC)
        
        # Base queryset for intrusion detections
        intrusion_query = Detection.objects.filter(
            detection_type='intrusion',
            timestamp__gte=start_time_utc,
            timestamp__lte=end_time_utc
        ).select_related('camera')
        
        # Apply filters
        if camera_filter:
            intrusion_query = intrusion_query.filter(camera_id=camera_filter)
            logger.info(f"Applied camera filter {camera_filter} to intrusion_query")
        
        # Get all intrusion checks for severity/type analysis
        intrusion_checks_query = intrusionChecks.objects.filter(
            intrusion=True,
            Detectionid__timestamp__gte=start_time_utc,
            Detectionid__timestamp__lte=end_time_utc
        ).select_related('camera', 'Detectionid')
        
        if camera_filter:
            intrusion_checks_query = intrusion_checks_query.filter(camera_id=camera_filter)
            logger.info(f"Applied camera filter {camera_filter} to intrusion_checks_query")
        
        if severity_filter:
            intrusion_checks_query = intrusion_checks_query.filter(severity=severity_filter)
        
        # Executive Summary Data
        total_incidents = intrusion_query.count()
        critical_incidents = intrusion_checks_query.filter(severity='critical').count()
        # Count pending alerts by checking Alert model directly
        pending_alerts = Alert.objects.filter(
            detection__in=intrusion_query,
            is_acknowledged=False
        ).count()
        
        # Debug logging for final counts
        logger.info(f"Final counts - total_incidents: {total_incidents}, critical_incidents: {critical_incidents}, pending_alerts: {pending_alerts}")
        
        # Timeline Data (24 hours for today, daily for longer periods)
        timeline_labels = []
        timeline_data = []
        
        if date_range == 'today':
            # Hourly breakdown for today
            for hour in range(24):
                hour_start = start_time.replace(hour=hour, minute=0, second=0, microsecond=0)
                hour_end = hour_start + timedelta(hours=1)
                hour_start_utc = hour_start.astimezone(pytz.UTC)
                hour_end_utc = hour_end.astimezone(pytz.UTC)
                
                count = intrusion_query.filter(
                    timestamp__gte=hour_start_utc,
                    timestamp__lt=hour_end_utc
                ).count()
                
                timeline_labels.append(f"{hour:02d}:00")
                timeline_data.append(count)
        else:
            # Daily breakdown for longer periods
            current_date = start_time.date()
            end_date = end_time.date()
            
            while current_date <= end_date:
                day_start = local_tz.localize(datetime.combine(current_date, datetime.min.time()))
                day_end = day_start + timedelta(days=1)
                day_start_utc = day_start.astimezone(pytz.UTC)
                day_end_utc = day_end.astimezone(pytz.UTC)
                
                count = intrusion_query.filter(
                    timestamp__gte=day_start_utc,
                    timestamp__lt=day_end_utc
                ).count()
                
                timeline_labels.append(current_date.strftime('%m/%d'))
                timeline_data.append(count)
                current_date += timedelta(days=1)
        
        # Severity Distribution
        severity_data = []
        severity_labels = ['Critical', 'High', 'Medium', 'Low']
        for severity in ['critical', 'high', 'medium', 'low']:
            count = intrusion_checks_query.filter(severity=severity).count()
            severity_data.append(count)
        
        # Location/Camera Analysis
        location_stats = intrusion_query.values('camera__name', 'camera__location').annotate(
            incident_count=Count('id')
        ).order_by('-incident_count')[:10]
        
        location_labels = []
        location_data = []
        for stat in location_stats:
            camera_name = stat['camera__name'] or f"Camera {stat['camera__id']}"
            location = stat['camera__location']
            label = f"{camera_name}" + (f" ({location})" if location else "")
            location_labels.append(label)
            location_data.append(stat['incident_count'])
        
        # Intrusion Type Analysis
        type_stats = intrusion_checks_query.values('intrusion_type').annotate(
            count=Count('id')
        ).order_by('-count')
        
        type_labels = []
        type_data = []
        for stat in type_stats:
            intrusion_type = stat['intrusion_type'] or 'Unknown'
            type_labels.append(intrusion_type.title())
            type_data.append(stat['count'])
        
        # If no real data, provide empty data to show no chart
        if not type_labels:
            type_labels = ['No Data']
            type_data = [0]
        
        # Hourly Pattern Analysis (for all data regardless of date range)
        hourly_pattern = [0] * 24
        for detection in intrusion_query:
            local_time = detection.timestamp.astimezone(local_tz)
            hourly_pattern[local_time.hour] += 1
        
        # Get detailed incidents for the table
        incidents = []
        for detection in intrusion_query.order_by('-timestamp')[:50]:  # Latest 50 incidents
            try:
                # Try to get intrusion check details
                intrusion_check = intrusionChecks.objects.filter(
                    Detectionid=detection
                ).first()
                
                # Try to get alert using reverse lookup
                try:
                    alert = Alert.objects.filter(detection=detection).first()
                except:
                    alert = None
                
                incident_data = {
                    'id': detection.id,
                    'timestamp': detection.timestamp.astimezone(local_tz),
                    'camera': detection.camera,
                    'detection_type': detection.detection_type,
                    'object_count': detection.object_count,
                    'annotated_snapshot_path': detection.annotated_snapshot_path,
                    'severity': intrusion_check.severity if intrusion_check else 'medium',
                    'intrusion_type': intrusion_check.intrusion_type if intrusion_check else None,
                    'alert': alert,
                    'response_time': None  # Could calculate if needed
                }
                incidents.append(incident_data)
            except Exception as e:
                logger.error(f"Error processing incident {detection.id}: {e}")
                continue
        
        # Performance metrics (mock data for now)
        performance_metrics = {
            'detection_accuracy': 94.2,
            'false_positive_rate': 5.8,
            'camera_uptime': 98.5,
            'resolution_rate': 87.3
        }
        
        # Prepare report data
        report_data = {
            'total_incidents': total_incidents,
            'critical_incidents': critical_incidents,
            'avg_response_time': 3.2,  # Mock data
            'pending_alerts': pending_alerts,
            'timeline_labels': json.dumps(timeline_labels),
            'timeline_data': json.dumps(timeline_data),
            'severity_labels': json.dumps(severity_labels),
            'severity_data': json.dumps(severity_data),
            'location_labels': json.dumps(location_labels),
            'location_data': json.dumps(location_data),
            'type_labels': json.dumps(type_labels),
            'type_data': json.dumps(type_data),
            'hourly_pattern': json.dumps(hourly_pattern),
            **performance_metrics
        }
        
        # Get all cameras for filter dropdown
        cameras = Camera.objects.filter(is_active=True)
        
        context = {
            'report_data': report_data,
            'incidents': incidents,
            'cameras': cameras,
            'current_filters': {
                'date_range': date_range,
                'camera_filter': camera_filter,
                'severity_filter': severity_filter
            }
        }
        
        return render(request, 'intrusion/intrusion_report.html', context)
        
    except Exception as e:
        logger.error(f"Error in intrusion_report: {e}")
        return HttpResponse(f"System error: {str(e)}", status=500)


@login_required(login_url='/login/')
@csrf_exempt
def api_latest_incidents(request, camera_id):
    """API endpoint for latest incident snapshots"""
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        
        # Get timezone from settings
        local_tz = pytz.timezone(settings.TIME_ZONE)
        
        # Get latest 5 intrusion incidents with snapshots
        latest_incidents = Detection.objects.filter(
            camera_id=camera_id,
            detection_type='intrusion',
            annotated_snapshot_path__isnull=False
        ).exclude(
            annotated_snapshot_path__exact=''
        ).order_by('-timestamp')[:5]
        
        incidents_data = []
        for detection in latest_incidents:
            try:
                # Get intrusion check details
                intrusion_check = intrusionChecks.objects.filter(
                    Detectionid=detection
                ).first()
                
                # Build snapshot URL
                snapshot_url = None
                if detection.annotated_snapshot_path:
                    if detection.annotated_snapshot_path.startswith('/'):
                        snapshot_url = detection.annotated_snapshot_path
                    else:
                        snapshot_url = f"/media/{detection.annotated_snapshot_path}"
                
                incident_data = {
                    'id': detection.id,
                    'timestamp': detection.timestamp.astimezone(local_tz).strftime('%Y-%m-%d %H:%M:%S'),
                    'timestamp_relative': get_relative_time(detection.timestamp),
                    'camera_name': detection.camera.name,
                    'severity': intrusion_check.severity if intrusion_check else 'medium',
                    'intrusion_type': intrusion_check.intrusion_type if intrusion_check else 'unauthorized_entry',
                    'object_count': detection.object_count or 1,
                    'snapshot_url': snapshot_url,
                    'description': detection.description[:100] + '...' if detection.description and len(detection.description) > 100 else detection.description
                }
                incidents_data.append(incident_data)
            except Exception as e:
                logger.error(f"Error processing incident {detection.id}: {e}")
                continue
        
        return JsonResponse({
            'incidents': incidents_data,
            'total_count': len(incidents_data)
        })
        
    except Exception as e:
        logger.error(f"Error in api_latest_incidents: {e}")
        return JsonResponse({'error': str(e)}, status=500)


def get_relative_time(timestamp):
    """Get relative time string like '2 hours ago'"""
    try:
        now = timezone.now()
        diff = now - timestamp
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "Just now"
    except:
        return "Unknown"


@login_required(login_url='/login/')
@csrf_exempt
def api_test(request):
    """Test API endpoint for connectivity"""
    # Also check if we have any intrusion data
    total_detections = Detection.objects.filter(detection_type='intrusion').count()
    total_intrusion_checks = intrusionChecks.objects.filter(intrusion=True).count()
    
    return JsonResponse({
        'status': 'success',
        'message': 'Intrusion API is working',
        'timestamp': timezone.now().isoformat(),
        'debug_info': {
            'total_intrusion_detections': total_detections,
            'total_intrusion_checks': total_intrusion_checks
        }
    })