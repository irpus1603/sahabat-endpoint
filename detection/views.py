from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
import logging
from core.models import Camera, SystemConfig


# Create your views here.

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