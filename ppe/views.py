from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from django.db.models import Count, Q, Avg, Case, When, IntegerField, F
from datetime import datetime, timedelta
import pytz
import csv
from django.http import HttpResponse
from core.models import Camera, SystemConfig, Detection, PPEChecks
from . import ppe_detection as ppe
import json

import logging


# configure logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Create your views here.

@login_required(login_url='/login/')
def list_ppe_camera(request):
    """View to display list of cameras with PPE detection capabilities"""
    try:
        # Get all active cameras
        cameras = Camera.objects.filter(is_active=True)
        
        # Calculate statistics
        total_cameras = cameras.count()
        ppe_enabled_cameras = cameras.filter(is_active=True).count()  # For now, all active cameras
        online_cameras = cameras.filter(is_active=True).count()
        total_violations = 0  # This would come from Detection model in future
        
        # Add mock compliance rates and violation counts for display
        for camera in cameras:
            camera.compliance_rate = 85  # Mock data - would be calculated from actual detections
            camera.violations_count = 0  # Mock data - would be from Detection model
            camera.location = camera.name  # Using name as location for now
        
        context = {
            'cameras': cameras,
            'total_cameras': total_cameras,
            'ppe_enabled_cameras': ppe_enabled_cameras,
            'online_cameras': online_cameras,
            'total_violations': total_violations,
        }
        
        return render(request, 'ppe/list_ppe_camera.html', context)
        
    except Exception as e:
        logger.error(f"Error in list_ppe_camera: {e}")
        return HttpResponse(f"System error: {str(e)}", status=500)

@login_required(login_url='/login/')
def ppe_detection_page(request, camera_id):
    """View to display PPE detection page with template"""
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        
        # Check if camera is active
        if not camera.is_active:
            logger.warning(f"Attempted to access inactive camera {camera.name}")
            return HttpResponse("Camera is not active", status=400)
        
        # Get violation data for the last 24 hours
        violation_data = get_violation_chart_data(camera_id)
        
        context = {
            'camera': camera,
            'current_time': timezone.now(),
            'violation_data': violation_data,
            'violation_data_json': json.dumps(violation_data),
        }
        
        return render(request, 'ppe/detection_ppe.html', context)
        
    except Camera.DoesNotExist:
        logger.error(f"Camera with ID {camera_id} not found")
        return HttpResponse("Camera not found", status=404)
    except Exception as e:
        logger.error(f"Error in ppe_detection_page for camera {camera_id}: {e}")
        return HttpResponse(f"System error: {str(e)}", status=500)

def get_violation_chart_data(camera_id):
    """Get PPE violation data for chart display - Last 6 hours with 30-minute intervals"""
    try:
        # Get current time in UTC (Django default)
        now_utc = timezone.now()
        
        # Convert to Jakarta timezone for display and calculation
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        now_jakarta = now_utc.astimezone(jakarta_tz)
        
        # Round up to next hour to include current period
        end_time_jakarta = (now_jakarta.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
        
        start_time_jakarta = end_time_jakarta - timedelta(hours=6)
        
        # Convert back to UTC for database queries
        end_time = end_time_jakarta.astimezone(pytz.UTC)
        start_time = start_time_jakarta.astimezone(pytz.UTC)
        
        logger.info(f"Generating violation data from {start_time_jakarta} to {end_time_jakarta} (Jakarta time)")
        logger.info(f"UTC times for DB query: {start_time} to {end_time}")
        
        # Initialize 30-minute interval data (6 hours = 12 intervals of 30 minutes each)
        violation_counts = []
        labels = []
        
        # Create 30-minute intervals for the last 6 hours (12 intervals)
        for i in range(12):  # 12 intervals of 30 minutes each
            interval_start_jakarta = start_time_jakarta + timedelta(minutes=i*30)
            interval_end_jakarta = interval_start_jakarta + timedelta(minutes=30)
            
            # Convert to UTC for database query
            interval_start_utc = interval_start_jakarta.astimezone(pytz.UTC)
            interval_end_utc = interval_end_jakarta.astimezone(pytz.UTC)
            
            # Create clear time labels using Jakarta time (e.g., "21:00", "21:30")
            time_label = interval_start_jakarta.strftime('%H:%M')
            labels.append(time_label)
            
            # Count violations in this 30-minute window using UTC times for DB query
            # PPE violation = any PPEChecks where at least one PPE item is missing
            violations = PPEChecks.objects.filter(
                camera_id=camera_id,
                Detectionid__timestamp__gte=interval_start_utc,
                Detectionid__timestamp__lt=interval_end_utc
            ).filter(
                Q(has_helmet=False) | Q(has_vest=False) | Q(has_goggles=False) | 
                Q(has_gloves=False) | Q(has_shoes=False)
            ).count()
            
            violation_counts.append(violations)
            logger.info(f"Time period {time_label}-{interval_end_jakarta.strftime('%H:%M')} (Jakarta): {violations} violations")
        
        logger.info(f"Final data - Labels: {labels}, Counts: {violation_counts}")
        
        return {
            'labels': labels,
            'data': violation_counts
        }
    except Exception as e:
        logger.error(f"Error getting violation chart data: {e}")
        # Return mock data with proper 30-minute intervals for last 6 hours (Jakarta time)
        now_utc = timezone.now()
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        now_jakarta = now_utc.astimezone(jakarta_tz)
        
        mock_labels = []
        for i in range(12):  # 12 intervals of 30 minutes
            time_point = now_jakarta - timedelta(hours=6) + timedelta(minutes=i*30)
            mock_labels.append(time_point.strftime('%H:%M'))
        
        return {
            'labels': mock_labels,
            'data': [2, 1, 0, 3, 1, 0, 2, 4, 1, 2, 0, 1]
        }

def api_violation_data(request, camera_id):
    """API endpoint to get violation data for charts"""
    try:
        logger.info(f"API call received for camera {camera_id}")
        violation_data = get_violation_chart_data(camera_id)
        logger.info(f"Returning violation data: {violation_data}")
        return JsonResponse(violation_data)
    except Exception as e:
        logger.error(f"Error in api_violation_data: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def api_test(request):
    """Simple test endpoint to verify API is working"""
    return JsonResponse({'status': 'API is working', 'message': 'PPE API endpoint test successful'})

def api_violation_summary(request, camera_id):
    """API endpoint to get violation summary for different time periods"""
    try:
        logger.info(f"API call received for violation summary camera {camera_id}")
        
        # Get current time in UTC and Jakarta timezone
        now_utc = timezone.now()
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        now_jakarta = now_utc.astimezone(jakarta_tz)
        
        # Calculate time periods
        end_time = now_utc
        start_1day = end_time - timedelta(days=1)
        start_7days = end_time - timedelta(days=7) 
        start_30days = end_time - timedelta(days=30)
        
        # Count violations for each period
        violations_1day = PPEChecks.objects.filter(
            camera_id=camera_id,
            Detectionid__timestamp__gte=start_1day,
            Detectionid__timestamp__lte=end_time
        ).filter(
            Q(has_helmet=False) | Q(has_vest=False) | Q(has_goggles=False) | 
            Q(has_gloves=False) | Q(has_shoes=False)
        ).count()
        
        violations_7days = PPEChecks.objects.filter(
            camera_id=camera_id,
            Detectionid__timestamp__gte=start_7days,
            Detectionid__timestamp__lte=end_time
        ).filter(
            Q(has_helmet=False) | Q(has_vest=False) | Q(has_goggles=False) | 
            Q(has_gloves=False) | Q(has_shoes=False)
        ).count()
        
        violations_30days = PPEChecks.objects.filter(
            camera_id=camera_id,
            Detectionid__timestamp__gte=start_30days,
            Detectionid__timestamp__lte=end_time
        ).filter(
            Q(has_helmet=False) | Q(has_vest=False) | Q(has_goggles=False) | 
            Q(has_gloves=False) | Q(has_shoes=False)
        ).count()
        
        summary_data = {
            'violations_1day': violations_1day,
            'violations_7days': violations_7days,
            'violations_30days': violations_30days,
            'last_updated': now_jakarta.strftime('%H:%M'),
            'current_time': now_jakarta.isoformat()
        }
        
        logger.info(f"Violation summary data: {summary_data}")
        return JsonResponse(summary_data)
        
    except Exception as e:
        logger.error(f"Error in api_violation_summary: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def api_equipment_violations(request, camera_id):
    """API endpoint to get PPE equipment violation breakdown for last 7 days"""
    try:
        logger.info(f"API call received for equipment violations camera {camera_id}")
        
        # Get current time in UTC
        now_utc = timezone.now()
        start_7days = now_utc - timedelta(days=7)
        
        # Count violations for each PPE equipment type in the last 7 days
        helmet_violations = PPEChecks.objects.filter(
            camera_id=camera_id,
            Detectionid__timestamp__gte=start_7days,
            Detectionid__timestamp__lte=now_utc,
            has_helmet=False
        ).count()
        
        vest_violations = PPEChecks.objects.filter(
            camera_id=camera_id,
            Detectionid__timestamp__gte=start_7days,
            Detectionid__timestamp__lte=now_utc,
            has_vest=False
        ).count()
        
        goggles_violations = PPEChecks.objects.filter(
            camera_id=camera_id,
            Detectionid__timestamp__gte=start_7days,
            Detectionid__timestamp__lte=now_utc,
            has_goggles=False
        ).count()
        
        gloves_violations = PPEChecks.objects.filter(
            camera_id=camera_id,
            Detectionid__timestamp__gte=start_7days,
            Detectionid__timestamp__lte=now_utc,
            has_gloves=False
        ).count()
        
        shoes_violations = PPEChecks.objects.filter(
            camera_id=camera_id,
            Detectionid__timestamp__gte=start_7days,
            Detectionid__timestamp__lte=now_utc,
            has_shoes=False
        ).count()
        
        equipment_data = {
            'labels': ['Helmet', 'Vest', 'Goggles', 'Gloves', 'Shoes'],
            'data': [helmet_violations, vest_violations, goggles_violations, gloves_violations, shoes_violations]
        }
        
        logger.info(f"Equipment violations data: {equipment_data}")
        return JsonResponse(equipment_data)
        
    except Exception as e:
        logger.error(f"Error in api_equipment_violations: {e}")
        # Return mock data on error
        return JsonResponse({
            'labels': ['Helmet', 'Vest', 'Goggles', 'Gloves', 'Shoes'],
            'data': [12, 8, 5, 3, 7]
        })

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
                for chunk in ppe.start_detection_stream(camera, hls_url):
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
def ppe_reports(request):
    """Main PPE reports page"""
    try:
        cameras = Camera.objects.filter(is_active=True)
        
        context = {
            'cameras': cameras,
            'current_time': timezone.now(),
        }
        
        return render(request, 'ppe/ppe_reports.html', context)
        
    except Exception as e:
        logger.error(f"Error in ppe_reports: {e}")
        return HttpResponse(f"System error: {str(e)}", status=500)

def get_date_range_from_request(request):
    """Helper function to get date range from request parameters"""
    period = request.GET.get('period', '1')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    
    now_utc = timezone.now()
    
    if start_date and end_date and period == 'custom':
        # Custom date range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Convert to timezone-aware datetimes
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        start_dt = jakarta_tz.localize(start_dt.replace(hour=0, minute=0, second=0))
        end_dt = jakarta_tz.localize(end_dt.replace(hour=23, minute=59, second=59))
        
        # Convert to UTC for database queries
        start_utc = start_dt.astimezone(pytz.UTC)
        end_utc = end_dt.astimezone(pytz.UTC)
    else:
        # Preset periods
        try:
            days = int(period)
        except (ValueError, TypeError):
            days = 1
            
        end_utc = now_utc
        start_utc = end_utc - timedelta(days=days)
    
    return start_utc, end_utc

def api_reports_kpi(request):
    """API endpoint for KPI widgets"""
    try:
        camera_id = request.GET.get('camera')
        start_utc, end_utc = get_date_range_from_request(request)
        
        # Base queryset
        base_query = PPEChecks.objects.filter(
            Detectionid__timestamp__gte=start_utc,
            Detectionid__timestamp__lte=end_utc
        )
        
        if camera_id:
            base_query = base_query.filter(camera_id=camera_id)
        
        # Total detections
        total_detections = base_query.count()
        
        # Total violations (any PPE missing)
        total_violations = base_query.filter(
            Q(has_helmet=False) | Q(has_vest=False) | Q(has_goggles=False) | 
            Q(has_gloves=False) | Q(has_shoes=False)
        ).count()
        
        # Compliance rate
        compliance_rate = 0
        if total_detections > 0:
            compliant_detections = total_detections - total_violations
            compliance_rate = round((compliant_detections / total_detections) * 100, 1)
        
        # Active cameras
        active_cameras = Camera.objects.filter(is_active=True).count()
        
        data = {
            'compliance_rate': compliance_rate,
            'total_detections': total_detections,
            'total_violations': total_violations,
            'active_cameras': active_cameras
        }
        
        return JsonResponse(data)
        
    except Exception as e:
        logger.error(f"Error in api_reports_kpi: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def api_reports_compliance_trend(request):
    """API endpoint for compliance rate trend chart"""
    try:
        camera_id = request.GET.get('camera')
        start_utc, end_utc = get_date_range_from_request(request)
        
        # Determine interval based on date range
        date_diff = (end_utc - start_utc).days
        if date_diff <= 1:
            # Hourly intervals for 1 day
            interval_hours = 2
            intervals = 12
        elif date_diff <= 7:
            # Daily intervals for 7 days
            interval_hours = 24
            intervals = date_diff
        else:
            # Weekly intervals for longer periods
            interval_hours = 24 * 7
            intervals = date_diff // 7
        
        labels = []
        compliance_rates = []
        
        for i in range(intervals):
            interval_start = start_utc + timedelta(hours=i * interval_hours)
            interval_end = start_utc + timedelta(hours=(i + 1) * interval_hours)
            
            # Create label
            jakarta_tz = pytz.timezone('Asia/Jakarta')
            interval_start_jakarta = interval_start.astimezone(jakarta_tz)
            
            if interval_hours < 24:
                labels.append(interval_start_jakarta.strftime('%H:%M'))
            elif interval_hours == 24:
                labels.append(interval_start_jakarta.strftime('%m/%d'))
            else:
                labels.append(interval_start_jakarta.strftime('%m/%d'))
            
            # Query data for this interval
            base_query = PPEChecks.objects.filter(
                Detectionid__timestamp__gte=interval_start,
                Detectionid__timestamp__lt=interval_end
            )
            
            if camera_id:
                base_query = base_query.filter(camera_id=camera_id)
            
            total = base_query.count()
            violations = base_query.filter(
                Q(has_helmet=False) | Q(has_vest=False) | Q(has_goggles=False) | 
                Q(has_gloves=False) | Q(has_shoes=False)
            ).count()
            
            if total > 0:
                compliance_rate = round(((total - violations) / total) * 100, 1)
            else:
                compliance_rate = 0
            
            compliance_rates.append(compliance_rate)
        
        data = {
            'labels': labels,
            'compliance_rates': compliance_rates
        }
        
        return JsonResponse(data)
        
    except Exception as e:
        logger.error(f"Error in api_reports_compliance_trend: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def api_reports_violation_types(request):
    """API endpoint for violation types donut chart"""
    try:
        camera_id = request.GET.get('camera')
        start_utc, end_utc = get_date_range_from_request(request)
        
        base_query = PPEChecks.objects.filter(
            Detectionid__timestamp__gte=start_utc,
            Detectionid__timestamp__lte=end_utc
        )
        
        if camera_id:
            base_query = base_query.filter(camera_id=camera_id)
        
        # Count violations by type
        helmet_violations = base_query.filter(has_helmet=False).count()
        vest_violations = base_query.filter(has_vest=False).count()
        goggles_violations = base_query.filter(has_goggles=False).count()
        gloves_violations = base_query.filter(has_gloves=False).count()
        shoes_violations = base_query.filter(has_shoes=False).count()
        
        data = {
            'labels': ['Helmet', 'Vest', 'Goggles', 'Gloves', 'Shoes'],
            'values': [helmet_violations, vest_violations, goggles_violations, gloves_violations, shoes_violations]
        }
        
        return JsonResponse(data)
        
    except Exception as e:
        logger.error(f"Error in api_reports_violation_types: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def api_reports_violation_heatmap(request):
    """API endpoint for violations by hour heatmap"""
    try:
        camera_id = request.GET.get('camera')
        start_utc, end_utc = get_date_range_from_request(request)
        
        base_query = PPEChecks.objects.filter(
            Detectionid__timestamp__gte=start_utc,
            Detectionid__timestamp__lte=end_utc
        ).filter(
            Q(has_helmet=False) | Q(has_vest=False) | Q(has_goggles=False) | 
            Q(has_gloves=False) | Q(has_shoes=False)
        )
        
        if camera_id:
            base_query = base_query.filter(camera_id=camera_id)
        
        # Group by hour of day
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        hours = []
        series = []
        
        for hour in range(24):
            hour_str = f"{hour:02d}:00"
            hours.append(hour_str)
            
            # Count violations for this hour across all days in the period
            violations_count = 0
            current_date = start_utc.date()
            end_date = end_utc.date()
            
            while current_date <= end_date:
                # Create datetime for this hour on this date (in Jakarta timezone)
                hour_start = jakarta_tz.localize(datetime.combine(current_date, datetime.min.time().replace(hour=hour)))
                hour_end = hour_start + timedelta(hours=1)
                
                # Convert to UTC for query
                hour_start_utc = hour_start.astimezone(pytz.UTC)
                hour_end_utc = hour_end.astimezone(pytz.UTC)
                
                count = base_query.filter(
                    Detectionid__timestamp__gte=hour_start_utc,
                    Detectionid__timestamp__lt=hour_end_utc
                ).count()
                
                violations_count += count
                current_date += timedelta(days=1)
            
            series.append(violations_count)
        
        data = {
            'hours': hours,
            'series': series
        }
        
        return JsonResponse(data)
        
    except Exception as e:
        logger.error(f"Error in api_reports_violation_heatmap: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def api_reports_location_compliance(request):
    """API endpoint for compliance by camera location"""
    try:
        start_utc, end_utc = get_date_range_from_request(request)
        
        cameras = Camera.objects.filter(is_active=True)
        camera_names = []
        compliance_rates = []
        
        for camera in cameras:
            base_query = PPEChecks.objects.filter(
                camera=camera,
                Detectionid__timestamp__gte=start_utc,
                Detectionid__timestamp__lte=end_utc
            )
            
            total = base_query.count()
            violations = base_query.filter(
                Q(has_helmet=False) | Q(has_vest=False) | Q(has_goggles=False) | 
                Q(has_gloves=False) | Q(has_shoes=False)
            ).count()
            
            if total > 0:
                compliance_rate = round(((total - violations) / total) * 100, 1)
            else:
                compliance_rate = 0
            
            camera_names.append(camera.name)
            compliance_rates.append(compliance_rate)
        
        data = {
            'cameras': camera_names,
            'compliance_rates': compliance_rates
        }
        
        return JsonResponse(data)
        
    except Exception as e:
        logger.error(f"Error in api_reports_location_compliance: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def api_reports_trend_analysis(request):
    """API endpoint for trend analysis - compliant vs non-compliant over time"""
    try:
        camera_id = request.GET.get('camera')
        start_utc, end_utc = get_date_range_from_request(request)
        
        # Determine interval based on date range
        date_diff = (end_utc - start_utc).days
        if date_diff <= 7:
            # Daily intervals for 7 days or less
            interval_hours = 24
            intervals = date_diff if date_diff > 0 else 1
        else:
            # Weekly intervals for longer periods
            interval_hours = 24 * 7
            intervals = date_diff // 7
        
        labels = []
        compliant = []
        violations = []
        
        for i in range(intervals):
            interval_start = start_utc + timedelta(hours=i * interval_hours)
            interval_end = start_utc + timedelta(hours=(i + 1) * interval_hours)
            
            # Create label
            jakarta_tz = pytz.timezone('Asia/Jakarta')
            interval_start_jakarta = interval_start.astimezone(jakarta_tz)
            
            if interval_hours == 24:
                labels.append(interval_start_jakarta.strftime('%m/%d'))
            else:
                labels.append(interval_start_jakarta.strftime('%m/%d'))
            
            # Query data for this interval
            base_query = PPEChecks.objects.filter(
                Detectionid__timestamp__gte=interval_start,
                Detectionid__timestamp__lt=interval_end
            )
            
            if camera_id:
                base_query = base_query.filter(camera_id=camera_id)
            
            total = base_query.count()
            violation_count = base_query.filter(
                Q(has_helmet=False) | Q(has_vest=False) | Q(has_goggles=False) | 
                Q(has_gloves=False) | Q(has_shoes=False)
            ).count()
            
            compliant_count = total - violation_count
            
            compliant.append(compliant_count)
            violations.append(violation_count)
        
        data = {
            'labels': labels,
            'compliant': compliant,
            'violations': violations
        }
        
        return JsonResponse(data)
        
    except Exception as e:
        logger.error(f"Error in api_reports_trend_analysis: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def api_reports_top_violators(request):
    """API endpoint for top violating cameras"""
    try:
        start_utc, end_utc = get_date_range_from_request(request)
        
        cameras = Camera.objects.filter(is_active=True)
        camera_data = []
        
        for camera in cameras:
            violations = PPEChecks.objects.filter(
                camera=camera,
                Detectionid__timestamp__gte=start_utc,
                Detectionid__timestamp__lte=end_utc
            ).filter(
                Q(has_helmet=False) | Q(has_vest=False) | Q(has_goggles=False) | 
                Q(has_gloves=False) | Q(has_shoes=False)
            ).count()
            
            camera_data.append({'name': camera.name, 'violations': violations})
        
        # Sort by violations count and take top 10
        camera_data.sort(key=lambda x: x['violations'], reverse=True)
        top_cameras = camera_data[:10]
        
        data = {
            'cameras': [c['name'] for c in top_cameras],
            'violations': [c['violations'] for c in top_cameras]
        }
        
        return JsonResponse(data)
        
    except Exception as e:
        logger.error(f"Error in api_reports_top_violators: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def api_reports_summary_table(request):
    """API endpoint for summary table data"""
    try:
        camera_id = request.GET.get('camera')
        start_utc, end_utc = get_date_range_from_request(request)
        
        cameras = Camera.objects.filter(is_active=True)
        if camera_id:
            cameras = cameras.filter(id=camera_id)
        
        summary_data = []
        
        for camera in cameras:
            base_query = PPEChecks.objects.filter(
                camera=camera,
                Detectionid__timestamp__gte=start_utc,
                Detectionid__timestamp__lte=end_utc
            )
            
            total_detections = base_query.count()
            violations = base_query.filter(
                Q(has_helmet=False) | Q(has_vest=False) | Q(has_goggles=False) | 
                Q(has_gloves=False) | Q(has_shoes=False)
            ).count()
            
            compliant = total_detections - violations
            compliance_rate = round((compliant / total_detections) * 100, 1) if total_detections > 0 else 0
            
            # Find most common violation
            violation_types = {
                'Helmet': base_query.filter(has_helmet=False).count(),
                'Vest': base_query.filter(has_vest=False).count(),
                'Goggles': base_query.filter(has_goggles=False).count(),
                'Gloves': base_query.filter(has_gloves=False).count(),
                'Shoes': base_query.filter(has_shoes=False).count(),
            }
            
            most_common_violation = max(violation_types.items(), key=lambda x: x[1])[0] if any(violation_types.values()) else 'None'
            
            # Last detection
            last_detection = base_query.order_by('-Detectionid__timestamp').first()
            last_detection_time = last_detection.Detectionid.timestamp.strftime('%Y-%m-%d %H:%M') if last_detection else 'No data'
            
            summary_data.append({
                'camera_name': camera.name,
                'total_detections': total_detections,
                'compliant': compliant,
                'violations': violations,
                'compliance_rate': compliance_rate,
                'most_common_violation': most_common_violation,
                'last_detection': last_detection_time
            })
        
        return JsonResponse(summary_data, safe=False)
        
    except Exception as e:
        logger.error(f"Error in api_reports_summary_table: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def api_reports_export(request):
    """Export full report as CSV"""
    try:
        camera_id = request.GET.get('camera')
        start_utc, end_utc = get_date_range_from_request(request)
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="ppe_report_{start_utc.date()}_to_{end_utc.date()}.csv"'
        
        writer = csv.writer(response)
        writer.writerow(['Timestamp', 'Camera', 'Track ID', 'Helmet', 'Vest', 'Goggles', 'Gloves', 'Shoes', 'Compliant'])
        
        base_query = PPEChecks.objects.filter(
            Detectionid__timestamp__gte=start_utc,
            Detectionid__timestamp__lte=end_utc
        ).select_related('camera', 'Detectionid')
        
        if camera_id:
            base_query = base_query.filter(camera_id=camera_id)
        
        for check in base_query.order_by('-Detectionid__timestamp'):
            is_compliant = all([check.has_helmet, check.has_vest, check.has_goggles, check.has_gloves, check.has_shoes])
            
            writer.writerow([
                check.Detectionid.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                check.camera.name,
                check.trackid or 'N/A',
                'Yes' if check.has_helmet else 'No',
                'Yes' if check.has_vest else 'No',
                'Yes' if check.has_goggles else 'No',
                'Yes' if check.has_gloves else 'No',
                'Yes' if check.has_shoes else 'No',
                'Yes' if is_compliant else 'No'
            ])
        
        return response
        
    except Exception as e:
        logger.error(f"Error in api_reports_export: {e}")
        return HttpResponse(f"Export error: {str(e)}", status=500)

def api_reports_export_table(request):
    """Export summary table as CSV"""
    try:
        camera_id = request.GET.get('camera')
        start_utc, end_utc = get_date_range_from_request(request)
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="ppe_summary_{start_utc.date()}_to_{end_utc.date()}.csv"'
        
        writer = csv.writer(response)
        writer.writerow(['Camera', 'Total Detections', 'Compliant', 'Violations', 'Compliance Rate (%)', 'Most Common Violation', 'Last Detection'])
        
        cameras = Camera.objects.filter(is_active=True)
        if camera_id:
            cameras = cameras.filter(id=camera_id)
        
        for camera in cameras:
            base_query = PPEChecks.objects.filter(
                camera=camera,
                Detectionid__timestamp__gte=start_utc,
                Detectionid__timestamp__lte=end_utc
            )
            
            total_detections = base_query.count()
            violations = base_query.filter(
                Q(has_helmet=False) | Q(has_vest=False) | Q(has_goggles=False) | 
                Q(has_gloves=False) | Q(has_shoes=False)
            ).count()
            
            compliant = total_detections - violations
            compliance_rate = round((compliant / total_detections) * 100, 1) if total_detections > 0 else 0
            
            # Find most common violation
            violation_types = {
                'Helmet': base_query.filter(has_helmet=False).count(),
                'Vest': base_query.filter(has_vest=False).count(),
                'Goggles': base_query.filter(has_goggles=False).count(),
                'Gloves': base_query.filter(has_gloves=False).count(),
                'Shoes': base_query.filter(has_shoes=False).count(),
            }
            
            most_common_violation = max(violation_types.items(), key=lambda x: x[1])[0] if any(violation_types.values()) else 'None'
            
            # Last detection
            last_detection = base_query.order_by('-Detectionid__timestamp').first()
            last_detection_time = last_detection.Detectionid.timestamp.strftime('%Y-%m-%d %H:%M') if last_detection else 'No data'
            
            writer.writerow([
                camera.name,
                total_detections,
                compliant,
                violations,
                compliance_rate,
                most_common_violation,
                last_detection_time
            ])
        
        return response
        
    except Exception as e:
        logger.error(f"Error in api_reports_export_table: {e}")
        return HttpResponse(f"Export error: {str(e)}", status=500)