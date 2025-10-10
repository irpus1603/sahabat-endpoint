from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.decorators import login_required
from core.models import Camera, Detection, PeopleCountingEvent
from django.shortcuts import redirect
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods
from django.http import StreamingHttpResponse
from core.models import SystemConfig
from . import people_counting as pcount
from django.utils import timezone
from django.db.models import Count, Q
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime, timedelta
from django.conf import settings
import pytz
import json

import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@login_required(login_url='/login/')
def list_people_counting_camera(request):
    """View to display list of cameras with People Counting capabilities"""
    try:
        # Get all active cameras
        cameras = Camera.objects.filter(is_active=True)
        
        # Calculate statistics
        total_cameras = cameras.count()
        people_counting_enabled_cameras = cameras.filter(is_active=True).count()  
        online_cameras = cameras.filter(is_active=True).count()
                
        # Add mock compliance rates and violation counts for display
        
        context = {
            'cameras': cameras,
            'people_counting_enabled_cameras': people_counting_enabled_cameras,
            'online_cameras': online_cameras,
            'total_cameras': total_cameras,
        }

        return render(request, 'peoplecounting/list_people_counting_camera.html', context)
        
    except Exception as e:
        logger.error(f"Error in list_people_counting_camera: {e}")
        return HttpResponse(f"System error: {str(e)}", status=500)


@login_required(login_url='/login/')
@require_http_methods(["GET"])
def people_counting_detection(request, camera_id):
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        camera_name = camera.name
        camera_source = camera.camera_source
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
                for chunk in pcount.start_detection_stream(camera, hls_url):
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
        logger.error(f"Error in detection_people_counting for camera {camera_id}: {e}")
        return HttpResponse(f"System error: {str(e)}", status=500)

@login_required(login_url='/login/')
def people_counting_detection_page(request, camera_id):
    """View to display People Counting detection page with template"""
    try:
        camera = get_object_or_404(Camera, id=camera_id)
        
        # Check if camera is active
        if not camera.is_active:
            logger.warning(f"Attempted to access inactive camera {camera.name}")
            return HttpResponse("Camera is not active", status=400)

        # Get people counting data for the last 24 hours
        people_counting_data = get_people_counting_chart_data(camera_id)

        context = {
            'camera': camera,
            'current_time': timezone.now(),
            'people_counting_data': people_counting_data,
            'people_counting_data_json': json.dumps(people_counting_data),
        }

        return render(request, 'peoplecounting/detection_people_counting.html', context)

    except Camera.DoesNotExist:
        logger.error(f"Camera with ID {camera_id} not found")
        return HttpResponse("Camera not found", status=404)
    except Exception as e:
        logger.error(f"Error in people_counting_detection_page for camera {camera_id}: {e}")
        return HttpResponse(f"System error: {str(e)}", status=500)

def get_people_counting_chart_data(camera_id):
    """Get people counting data for the last 6 hours with 30-minute intervals using PeopleCountingEvent table"""
    try:
        # Get timezone from settings
        local_tz = pytz.timezone(settings.TIME_ZONE)
        
        # Get current time in local timezone
        now_utc = timezone.now()
        now_local = now_utc.astimezone(local_tz)
        
        # Set time range (last 6 hours from current local time)
        end_time = now_local.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        start_time = end_time - timedelta(hours=6)
        
        # Convert back to UTC for database queries (Django stores in UTC)
        start_time_utc = start_time.astimezone(pytz.UTC)
        end_time_utc = end_time.astimezone(pytz.UTC)

        logger.info(f"Getting people counting data for camera {camera_id}")
        logger.info(f"Local time range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"UTC time range for query: {start_time_utc.strftime('%Y-%m-%d %H:%M')} to {end_time_utc.strftime('%Y-%m-%d %H:%M')}")
        
        # Generate 30-minute interval labels and time slots in local time
        labels = []
        time_slots_local = []
        current_time = start_time
        
        while current_time < end_time:
            labels.append(current_time.strftime('%H:%M'))
            time_slots_local.append(current_time)
            current_time += timedelta(minutes=30)
        
        # Convert local time slots to UTC for database queries
        time_slots_utc = []
        for slot in time_slots_local:
            time_slots_utc.append(slot.astimezone(pytz.UTC))
        
        # Get people counting data for each time slot using PeopleCountingEvent table
        in_data = []
        out_data = []
        total_events = 0
        
        for i in range(len(time_slots_utc) - 1):
            slot_start_utc = time_slots_utc[i]
            slot_end_utc = time_slots_utc[i + 1]
            slot_start_local = time_slots_local[i]
            slot_end_local = time_slots_local[i + 1]
            
            # Count IN events
            in_count = PeopleCountingEvent.objects.filter(
                camera_id=camera_id,
                direction='IN',
                timestamp__gte=slot_start_utc,
                timestamp__lt=slot_end_utc
            ).count()
            
            # Count OUT events
            out_count = PeopleCountingEvent.objects.filter(
                camera_id=camera_id,
                direction='OUT',
                timestamp__gte=slot_start_utc,
                timestamp__lt=slot_end_utc
            ).count()

            in_data.append(in_count)
            out_data.append(out_count)
            total_events += in_count + out_count

            logger.debug(f"Local time slot {slot_start_local.strftime('%H:%M')}-{slot_end_local.strftime('%H:%M')}: IN={in_count}, OUT={out_count}")
        
        logger.info(f"Total people counting events found: {total_events}")
        
        return {
            'labels': labels,
            'in_data': in_data,
            'out_data': out_data
        }
        
    except Exception as e:
        logger.error(f"Error getting people counting chart data: {e}")
        # Return empty data structure on error
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
            labels = ['08:00', '08:30', '09:00', '09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00', '13:30']
        
        return {
            'labels': labels,
            'in_data': [0] * len(labels),
            'out_data': [0] * len(labels)
        }

# API ENDPOINTS FOR PEOPLE COUNTING

@csrf_exempt
def api_test(request):
    """API test endpoint"""
    if request.method == 'GET':
        return JsonResponse({
            'status': 'success',
            'message': 'People counting API is working',
            'timestamp': timezone.now().isoformat()
        })
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def api_events(request, camera_id):
    """API endpoint for people counting events data (chart data)"""
    if request.method == 'GET':
        try:
            camera = get_object_or_404(Camera, id=camera_id)
            data = get_people_counting_chart_data(camera_id)
            return JsonResponse(data)
        except Exception as e:
            logger.error(f"Error in api_events: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def api_summary(request, camera_id):
    """API endpoint for people counting summary statistics"""
    if request.method == 'GET':
        try:
            camera = get_object_or_404(Camera, id=camera_id)
            
            # Get current time
            now = timezone.now()
            
            # Last 24 hours
            last_24h = now - timedelta(hours=24)
            people_in_1day = PeopleCountingEvent.objects.filter(
                camera_id=camera_id,
                direction='IN',
                timestamp__gte=last_24h
            ).count()
            
            people_out_1day = PeopleCountingEvent.objects.filter(
                camera_id=camera_id,
                direction='OUT',
                timestamp__gte=last_24h
            ).count()
            
            # Last 7 days
            last_7days = now - timedelta(days=7)
            people_in_7days = PeopleCountingEvent.objects.filter(
                camera_id=camera_id,
                direction='IN',
                timestamp__gte=last_7days
            ).count()
            
            people_out_7days = PeopleCountingEvent.objects.filter(
                camera_id=camera_id,
                direction='OUT',
                timestamp__gte=last_7days
            ).count()
            
            # Get latest occupancy from most recent event
            latest_event = PeopleCountingEvent.objects.filter(
                camera_id=camera_id
            ).order_by('-timestamp').first()
            
            current_occupancy = latest_event.occupancy if latest_event else 0
            todays_traffic = people_in_1day + people_out_1day
            
            return JsonResponse({
                'people_in_1day': people_in_1day,
                'people_out_1day': people_out_1day,
                'people_in_7days': people_in_7days,
                'people_out_7days': people_out_7days,
                'current_occupancy': current_occupancy,
                'todays_traffic': todays_traffic,
                'last_updated': now.strftime('%H:%M:%S')
            })
            
        except Exception as e:
            logger.error(f"Error in api_summary: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def api_daily_traffic(request, camera_id):
    """API endpoint for daily traffic data (last 7 days)"""
    if request.method == 'GET':
        try:
            camera = get_object_or_404(Camera, id=camera_id)
            
            # Get timezone from settings
            local_tz = pytz.timezone(settings.TIME_ZONE)
            now_local = timezone.now().astimezone(local_tz)
            
            # Get last 7 days
            labels = []
            in_data = []
            out_data = []
            
            for i in range(6, -1, -1):  # 7 days ago to today
                day = now_local - timedelta(days=i)
                day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
                day_end = day_start + timedelta(days=1)
                
                # Convert to UTC for database query
                day_start_utc = day_start.astimezone(pytz.UTC)
                day_end_utc = day_end.astimezone(pytz.UTC)
                
                # Count events for this day
                in_count = PeopleCountingEvent.objects.filter(
                    camera_id=camera_id,
                    direction='IN',
                    timestamp__gte=day_start_utc,
                    timestamp__lt=day_end_utc
                ).count()
                
                out_count = PeopleCountingEvent.objects.filter(
                    camera_id=camera_id,
                    direction='OUT',
                    timestamp__gte=day_start_utc,
                    timestamp__lt=day_end_utc
                ).count()
                
                labels.append(day.strftime('%a %m/%d'))  # Mon 12/25
                in_data.append(in_count)
                out_data.append(out_count)
            
            return JsonResponse({
                'labels': labels,
                'in_data': in_data,
                'out_data': out_data
            })
            
        except Exception as e:
            logger.error(f"Error in api_daily_traffic: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def api_hourly_distribution(request, camera_id):
    """API endpoint for hourly distribution data (last 7 days by hour)"""
    if request.method == 'GET':
        try:
            camera = get_object_or_404(Camera, id=camera_id)
            
            # Get timezone from settings
            local_tz = pytz.timezone(settings.TIME_ZONE)
            now_local = timezone.now().astimezone(local_tz)
            
            # Get last 7 days
            start_time = now_local - timedelta(days=7)
            start_time_utc = start_time.astimezone(pytz.UTC)
            end_time_utc = now_local.astimezone(pytz.UTC)
            
            # Create hourly buckets (24 hours)
            labels = []
            in_data = []
            out_data = []
            
            for hour in range(24):
                # Count events for this hour across all days in the period
                in_count = PeopleCountingEvent.objects.filter(
                    camera_id=camera_id,
                    direction='IN',
                    timestamp__gte=start_time_utc,
                    timestamp__lt=end_time_utc,
                    timestamp__hour=hour
                ).count()
                
                out_count = PeopleCountingEvent.objects.filter(
                    camera_id=camera_id,
                    direction='OUT',
                    timestamp__gte=start_time_utc,
                    timestamp__lt=end_time_utc,
                    timestamp__hour=hour
                ).count()
                
                labels.append(f"{hour:02d}:00")
                in_data.append(in_count)
                out_data.append(out_count)
            
            return JsonResponse({
                'labels': labels,
                'in_data': in_data,
                'out_data': out_data
            })
            
        except Exception as e:
            logger.error(f"Error in api_hourly_distribution: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Method not allowed'}, status=405)
