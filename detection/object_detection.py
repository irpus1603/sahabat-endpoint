
import cv2
import logging
import numpy as np
from threading import Thread
from ultralytics import YOLO
from core.models import SystemConfig
import time
from queue import Queue
from .send_notifications import  send_telegram_notification
from .save_snapshots import save_snapshot_and_record
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_time_sleep():
    """Get frame processing delay from config"""
    try:
        return float(SystemConfig.get_value('frame_proc_delay', 0.033))
    except:
        return 0.033  # Default 30 FPS

# Lazy loading for YOLO model
det_model = None
label_map = None

def get_model():
    global det_model, label_map
    if det_model is None:
        # Set environment variables to prevent fork crashes
        import os
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        det_model = YOLO("models/yolo11n.mlpackage", task='detect')
        label_map = det_model.names
    return det_model, label_map


# Queue for frames to be processed - Reduced size for better performance
frame_queue = Queue(maxsize=100) 

try:
    whitelist = SystemConfig.get_value('obj_det_cls_id_whitelist')
    if whitelist:
        list_class = [int(x.strip()) for x in whitelist.split(',')]
    else:
        list_class = (0,1,2,3)
except Exception as e:
    # Table doesn't exist yet (during migrations) or other error
    list_class = (0,1,2,3)

# Global timestamp tracking for notifications (prevents spam)
save_date_timestamp = {}

def is_blank_frame(frame, threshold=10):
    """Check if frame is blank/black"""
    if frame is None or frame.size == 0:
        return True
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stddev = np.std(gray_frame)
    return stddev < threshold

def is_uniform_frame(frame, threshold=15):
    """Check if frame has uniform/no content"""
    if frame is None or frame.size == 0:
        return True
    return np.all(np.abs(frame - frame.mean()) < threshold)

def test_video_source_connectivity(hls_url, timeout=15):
     """
     Test HLS video source connectivity without starting full capture
     Returns: dict with status information
     """
     result = {
         'connected': False,
         'error': None,
         'frame_readable': False,
         'source_info': {}
     }
     
     cap = None
     try:
         logger.info(f"Testing connectivity to HLS source: {hls_url}")
         
         # Initialize video capture with timeout (longer for HLS)
         if hls_url[0:5].lower() == 'local':
            cap = cv2.VideoCapture(int(hls_url[6]))
         else:
             cap = cv2.VideoCapture(hls_url, cv2.CAP_FFMPEG)
             cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
             cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # HLS may need more time
         
         if not cap.isOpened():
             result['error'] = "Failed to open video source"
             return result
         
         result['connected'] = True
         
         # Get source information
         try:
             result['source_info'] = {
                 'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 'fps': int(cap.get(cv2.CAP_PROP_FPS)),
                 'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC))
             }
         except:
             pass
         
         # Try to read a test frame (HLS may take longer to provide first frame)
         ret, frame = cap.read()
         if ret and frame is not None and not is_blank_frame(frame):
             result['frame_readable'] = True
             logger.info(f"HLS source test successful: {hls_url}")
         else:
             result['error'] = "Cannot read frames from HLS source"
             logger.warning(f"HLS source connected but frames not readable: {hls_url}")
             
     except Exception as e:
         result['error'] = str(e)
         logger.error(f"Video source test failed: {e}")
         
     finally:
         if cap is not None:
             try:
                 cap.release()
             except:
                 pass
                 
     return result

def create_robust_capture_with_monitoring(hls_url, frame_queue, status_callback=None, shared_status=None):

     connection_status = shared_status if shared_status else {
         'connected': False,
         'last_frame_time': None,
         'total_frames': 0,
         'connection_attempts': 0,
         'errors': []
     }
     
     def update_status(status_dict):
         connection_status.update(status_dict)
         if status_callback:
             try:
                 status_callback(connection_status.copy())
             except Exception as e:
                 logger.error(f"Status callback error: {e}")
     
     # Pre-test HLS connectivity
     test_result = test_video_source_connectivity(hls_url)
     if not test_result['connected']:
         logger.error(f"Initial HLS connectivity test failed: {test_result['error']}")
         update_status({'error': test_result['error']})
     
     # Start the enhanced capture with monitoring
     try:
         capture_video_with_monitoring(hls_url, frame_queue, update_status, connection_status)
     except Exception as e:
         logger.error(f"HLS capture monitoring failed: {e}")
         update_status({'error': str(e), 'connected': False})

def capture_video_with_monitoring(hls_url, frame_queue, status_update_func, connection_status):
    """Enhanced HLS capture function with status monitoring"""
    connection_attempts = 0
    max_attempts = 5
    base_retry_delay = 3  # Slightly longer for HLS
    consecutive_failures = 0
    last_frame_time = time.time()
    reconnect_threshold = 15  # Longer threshold for HLS due to segment nature
    total_frames = 0
    
    while True:
        # Check for stop signal
        if connection_status.get('stop_requested', False):
            logger.info("Stop signal received in capture thread, exiting...")
            break
            
        cap = None
        retry_delay = base_retry_delay  # Initialize retry_delay
        try:
            connection_attempts += 1
            status_update_func({
                'connection_attempts': connection_attempts,
                'status': 'connecting'
            })
            
            logger.info(f"Attempting to connect to HLS source (attempt {connection_attempts}): {hls_url}")
            
            # Initialize the video capture with HLS-optimized parameters
            if hls_url[0:5].lower() == 'local':
                cap = cv2.VideoCapture(int(hls_url[6]))
            else:
                cap = cv2.VideoCapture(hls_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)  # Longer for HLS
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000)   # Longer for HLS

            # HLS-optimized streaming settings
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)  # Smaller for HLS
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Try to set H.264 if available (common for HLS)
            try:
                if hasattr(cv2, 'VideoWriter_fourcc'):
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            except (AttributeError, Exception):
                logger.debug("H.264 codec setting not supported for HLS, using default")

            if not cap.isOpened():
                raise ConnectionError("Unable to open video stream")

            # Test initial frame read
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                raise ConnectionError("Unable to read initial frame from stream")

            # Success - update status
            connection_attempts = 0
            consecutive_failures = 0
            last_frame_time = time.time()
             
            status_update_func({
                'connected': True,
                'status': 'streaming',
                'last_frame_time': last_frame_time,
                'connection_attempts': 0
            })
            
            logger.info(f"Video stream connected successfully")
            
            # Put test frame in queue if valid
            if not is_blank_frame(test_frame) and not is_uniform_frame(test_frame):
                frame_queue.put(test_frame)
                total_frames += 1

            # Main streaming loop with enhanced monitoring
            frame_count = 0
            health_check_interval = 30
            
            while True:
                # Check for stop signal
                if connection_status.get('stop_requested', False):
                    logger.info("Stop signal received in streaming loop, exiting...")
                    return
                    
                current_time = time.time()
                
                # Timeout check
                if current_time - last_frame_time > reconnect_threshold:
                    raise ConnectionError(f"Frame timeout after {reconnect_threshold} seconds")
                
                # Read frame
                ret, frame = cap.read()
                
                # Check for stop signal immediately after frame read
                if connection_status.get('stop_requested', False):
                    logger.info("Stop signal received after frame read, exiting...")
                    return
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        raise ConnectionError("Multiple consecutive frame failures")
                    time.sleep(get_time_sleep())
                    continue
                
                # Success - reset counters and update status
                consecutive_failures = 0
                last_frame_time = current_time
                frame_count += 1
                total_frames += 1

                # Skip bad frames
                if is_blank_frame(frame) or is_uniform_frame(frame):
                    continue

                # Simple queue management
                current_queue_size = frame_queue.qsize()
                
                # Clear queue if it gets too full (for maxsize=100)
                if current_queue_size > 90:
                    # Clear to 40 frames to prevent lag
                    while frame_queue.qsize() > 60:
                        try:
                            frame_queue.get_nowait()
                        except:
                            break
                    logger.warning(f"Queue cleared: {current_queue_size} -> {frame_queue.qsize()}")

                # Add frame to queue
                try:
                    frame_queue.put(frame, block=False)
                except:
                    # Queue full - replace oldest frame
                    try:
                        frame_queue.get_nowait()
                        frame_queue.put(frame, block=False)
                    except:
                        continue
                
                # Update status periodically
                if frame_count % health_check_interval == 0:
                    status_update_func({
                        'total_frames': total_frames,
                        'last_frame_time': last_frame_time,
                        'queue_size': frame_queue.qsize()
                    })
                    
                    if not cap.isOpened():
                        raise ConnectionError("Video capture object closed")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[NETWORK] Video capture error (attempt {connection_attempts}/{max_attempts}): {error_msg}")
            
            # Update error status
            status_update_func({
                'connected': False,
                'status': 'error',
                'error': error_msg,
                'connection_attempts': connection_attempts
            })
            
            # Calculate retry delay
            if "timeout" in error_msg.lower() or "network" in error_msg.lower():
                retry_delay = min(base_retry_delay * (2 ** min(connection_attempts, 4)), 30)
            else:
                retry_delay = base_retry_delay * min(connection_attempts, 3)
            
            if connection_attempts >= max_attempts:
                logger.error(f"[NETWORK] Max connection attempts reached ({max_attempts}), waiting 60 seconds before reset...")
                retry_delay = 60
                connection_attempts = 0
            
            logger.info(f"[NETWORK] Retrying connection in {retry_delay} seconds... (attempt {connection_attempts})")
            
        finally:
            if cap is not None:
                try:
                    cap.release()
                except:
                    pass
                cap = None
            
            # Status update for disconnect
            status_update_func({
                'connected': False,
                'status': 'disconnected'
            })
            
            # Wait before retry
            delay = locals().get('retry_delay', base_retry_delay)
            time.sleep(delay)

def save_snapshot_async(*args):
    Thread(target=save_snapshot_and_record, args=args).start()

def send_telegram_async(*args):
    Thread(target=send_telegram_notification, args=args).start()

def start_detection_stream(camera, hls_url):
   
    # Create frame queue and connection status
    frame_queue = Queue(maxsize=100)
    connection_status = {'connected': False, 'error': None, 'stop_requested': False}
    
    def status_callback(status):
        connection_status.update(status)
        #logger.info(f"Camera {camera.name} status: {status.get('status', 'unknown')}")
    
    # Test connectivity first
    connectivity_test = test_video_source_connectivity(hls_url, timeout=10)
    if not connectivity_test['connected']:
        logger.warning(f"Initial connectivity test failed for camera {camera.name}: {connectivity_test['error']}")
        yield create_status_frame(f"Connection failed: {connectivity_test['error']}")
        return
    
    # Start capture thread
    capture_thread = Thread(
        target=create_robust_capture_with_monitoring,
        args=(hls_url, frame_queue, status_callback, connection_status),
        daemon=True
    )
    capture_thread.start()
    
    # Wait for capture to start
    import time
    time.sleep(1.0)
    
    # Check if capture started successfully
    if frame_queue.empty():
        time.sleep(get_time_sleep())
        if frame_queue.empty() and not connection_status.get('connected', False):
            error_msg = connection_status.get('error', 'Failed to start video capture')
            logger.error(f"Video capture failed to start for camera {camera.name}: {error_msg}")
            connection_status['stop_requested'] = True
            yield create_status_frame(f"Failed to start stream: {error_msg}")
            return
    
    logger.info(f"Starting detection stream for camera {camera.name}")
    
    try:
        # Stream processing with automatic cleanup
        for frame_data in process_video_stream_with_reconnect(frame_queue, camera, connection_status):
            yield frame_data
            
    except GeneratorExit:
        # Client disconnected - cleanup resources
        logger.info(f"Client disconnected for camera {camera.name}, cleaning up...")
        connection_status['stop_requested'] = True
        
        # Wait a moment for threads to clean up
        if capture_thread.is_alive():
            logger.info(f"Waiting for capture thread to stop for camera {camera.name}...")
            capture_thread.join(timeout=3.0)
            if capture_thread.is_alive():
                logger.warning(f"Capture thread for camera {camera.name} did not stop gracefully within 3 seconds")
            else:
                logger.info(f"Capture thread for camera {camera.name} stopped successfully")
        
        raise
    except Exception as e:
        logger.error(f"Detection stream error for camera {camera.name}: {e}")
        connection_status['stop_requested'] = True
        yield create_status_frame(f"Stream error: {str(e)}")

def process_video_stream_with_reconnect(frame_queue, camera, connection_status):
     """
     Enhanced video stream processing with connection monitoring
     """
     try:
         # Use the original process_video_stream but with connection monitoring
         for frame_data in process_video_stream(frame_queue, camera):
             # Check connection status periodically
             if not connection_status.get('connected', True):
                 # If disconnected, yield a status frame
                 status_msg = connection_status.get('error', 'Connection lost')
                 yield create_status_frame(f"Reconnecting... {status_msg}")
             else:
                 yield frame_data
                 
     except GeneratorExit:
         # Client disconnected - cleanup resources
         logger.info(f"Client disconnected for camera {camera.name}, cleaning up...")
         # Signal threads to stop by setting a stop flag
         connection_status['stop_requested'] = True
         raise
     except Exception as e:
         logger.error(f"Stream processing error: {e}")
         yield create_status_frame(f"Stream error: {str(e)}")

def create_status_frame(message):
     """Create a simple status frame with text message"""
     import cv2
     import numpy as np
     
     # Create a black frame
     frame = np.zeros((480, 640, 3), dtype=np.uint8)
     
     # Add status text
     font = cv2.FONT_HERSHEY_SIMPLEX
     font_scale = 1
     color = (0, 255, 255)  # Yellow
     thickness = 2
     # Calculate text size and position
     text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
     text_x = (frame.shape[1] - text_size[0]) // 2
     text_y = (frame.shape[0] + text_size[1]) // 2
     
     cv2.putText(frame, message, (text_x, text_y), font, font_scale, color, thickness)
     
    # Add timestamp
     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
     cv2.putText(frame, timestamp, (10, 30), font, 0.7, (255, 255, 255), 1)
     
     # Encode frame
     ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
     if ret:
         frame_bytes = buffer.tobytes()
         return (b'--frame\r\n'
                 b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
     else:
         return b''

# Function to check if a point is inside a polygon
def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# Function to get polygon coordinates from the database
def get_polygon_from_coordinates(coordinates_str):
    if not coordinates_str:
        return None
    coords = list(map(int, coordinates_str.split(',')))
    return np.array(coords, dtype=np.int32).reshape((-1, 1, 2))

def draw_bounding_box(frame, track_id, cls_name, confidence, x1, y1, x2, y2, color=(255, 0, 0), thickness=1):
    # Draw grey bounding box rectangle with thin line
    rect_color = (192, 192, 192)  # Grey color
    rect_thickness = 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, rect_thickness)
    
    label = f"[ID.{track_id}]{cls_name.capitalize()}:{confidence:.2f}"
    
    # Calculate text size and position
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.3
    font_thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    # Position for text
    text_x = x1
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
    
    # Draw grey background rectangle for text
    background_color = (255, 255, 255)  # White background
    padding = 2
    cv2.rectangle(frame, 
                  (text_x - padding, text_y - text_height - padding), 
                  (text_x + text_width + padding, text_y + baseline + padding), 
                  background_color, -1)
    
    # Draw text on top of white background
    text_color = (128, 128, 128)  # Black text
    cv2.putText(frame, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

def draw_roi_polylines(frame, polygons, transparency=0.6):
    """
    Draw ROI polylines with transparent gray fill
    
    Args:
        frame: Input frame to draw on
        polygons: List of polygon coordinates
        transparency: Transparency level (0.0 to 1.0)
    """
    if not polygons:
        return
    
    # Create overlay for transparency effect
    overlay = frame.copy()
    
    for i, polygon in enumerate(polygons):
        if polygon is not None and len(polygon) > 2:
            # Fill polygon with gray color on overlay
            fill_color = (128, 128, 240)  # Red color
            cv2.fillPoly(overlay, [polygon], fill_color)
            
            # Draw polyline border
            border_color = (96, 96, 96)  # Darker gray for border
            cv2.polylines(overlay, [polygon], isClosed=True, color=border_color, thickness=2)
    
    # Blend overlay with original frame for transparency
    cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0, frame)

def draw_detection_counter(frame, count, object_type="people", 
                          font=cv2.FONT_HERSHEY_DUPLEX, font_scale=0.4, 
                          font_thickness=1, text_color=(255, 255, 255), 
                          bg_color=(0, 0, 0), x_pos=50, y_offset=50):
    """
    Simple function to draw detection counter text on frame with background
    
    Args:
        frame: Input frame
        count: Number of detected objects
        object_type: Type of object (e.g., "people", "cars", "objects")
        font: OpenCV font type
        font_scale: Font size
        font_thickness: Text thickness
        text_color: BGR color tuple for text
        bg_color: BGR color tuple for background
        x_pos: X position
        y_offset: Y offset from bottom
    """
    frame_height = frame.shape[0]
    y_position = frame_height - y_offset
    
    text = f'Total {object_type} detected: {count}'
    
    # Calculate text size for background
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Draw white background rectangle
    padding = 5
    cv2.rectangle(frame, 
                  (x_pos - padding, y_position - text_height - padding), 
                  (x_pos + text_width + padding, y_position + baseline + padding), 
                  bg_color, -1)
    
    # Draw text on top of background
    cv2.putText(frame, text, (x_pos, y_position), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

# Function to process frames from the queue
def process_video_stream(frame_queue, camera): 
    global save_date_timestamp
    cameraid = camera.id
    cameraname = camera.name
    model, _ = get_model()
    
    
    # Get ROI polygons from both AI assignment and camera model
    polygons = []
    
    roi_coords = camera.get_roi_coordinates()
    #logger.info(f"Camera {cameraname} (ID: {cameraid}) - ROI coordinates loaded: {roi_coords}")
    
    if roi_coords is not None:
        
        if len(roi_coords.shape) == 2 and roi_coords.shape[1] == 2:
            # Convert (N, 2) to (N, 1, 2) format for OpenCV
            polygons = [roi_coords.reshape(-1, 1, 2)]
        else:
            polygons = [roi_coords] if len(roi_coords.shape) == 3 else [roi_coords.reshape(-1, 1, 2)]
        
    else:
        polygons = []
        logger.warning(f"Camera {cameraname} - No ROI coordinates found")

    
    
    previous_snapshot_time = 0.0

    try:
        while True:
                frame = frame_queue.get()  # Get frame from queue (blocking if empty)
                if frame is None:
                    logger.error("Cannot capture frame")
                    continue

                current_time_pass = datetime.now().strftime('%d-%m-%y %H:%M:%S')
                message_body = f"Security intrusion detected on camera {cameraname} at {current_time_pass}"
                
                # Resize frame for processing
                original_height, original_width = frame.shape[:2]
                target_width = 640
                target_height = original_height * target_width // original_width
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                
                # Calculate scaling factors for polygon coordinates
                scale_x = target_width / original_width
                scale_y = target_height / original_height

                # Draw ROI polygons if configured
                if len(polygons) > 0:
                    scaled_polygons = []
                    for polygon in polygons:
                        if polygon is not None:
                            # Scale polygon coordinates to match resized frame
                            scaled_polygon = polygon.copy()
                            scaled_polygon[:, 0, 0] = (scaled_polygon[:, 0, 0] * scale_x).astype(np.int32)
                            scaled_polygon[:, 0, 1] = (scaled_polygon[:, 0, 1] * scale_y).astype(np.int32)
                            scaled_polygons.append(scaled_polygon)
                    
                    # Draw ROI with transparent fill
                    draw_roi_polylines(frame, scaled_polygons, transparency=0.3)
                

                frame_height = frame.shape[0]
                # Calculate position for text
                x_position = 50  # Fixed padding from the left
                y_position = frame_height - (50)  # Padding from the bottom of the frame

                # Run AI detection and tracking
                results = model.track(source=frame, persist=True, conf=0.3, verbose=False)

                # Check if results contain boxes
                if results and len(results) > 0 and results[0].boxes is not None:
                    person_count = sum(1 for track in results[0].boxes if int(track.cls.item()) == 0)
                    # Use the new function to draw detection counter
                    draw_detection_counter(frame, person_count, "people")

                    for track in results[0].boxes:
                        x1, y1, x2, y2 = map(int, track.xyxy[0])
                        cls_id = int(track.cls.item())
                        confidence = track.conf.item()
                        track_id = int(track.id.item()) if hasattr(track, 'id') and track.id is not None else -1

                        if cls_id not in list_class:
                            continue
                        
                        center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                        
                        # Check if center point is inside any of the ROI polygons (using scaled coordinates)
                        point_inside_any_polygon = False
                        if len(polygons) > 0:
                            for polygon in polygons:
                                if polygon is not None:
                                    # Scale polygon coordinates to match resized frame for detection
                                    scaled_polygon = polygon.copy()
                                    scaled_polygon[:, 0, 0] = (scaled_polygon[:, 0, 0] * scale_x).astype(np.int32)
                                    scaled_polygon[:, 0, 1] = (scaled_polygon[:, 0, 1] * scale_y).astype(np.int32)
                                    
                                    if is_point_in_polygon(center_point, scaled_polygon):
                                        point_inside_any_polygon = True
                                        break
                        
                        if point_inside_any_polygon:
                            current_time = time.time()
                            # Enhanced security intrusion detection with proper timing

                            snapshot_timer_value = SystemConfig.get_value('snapshot_counter')
                            snapshot_timer = int(snapshot_timer_value) if snapshot_timer_value else 60
                            
                            if current_time - previous_snapshot_time >= snapshot_timer:
                                # Enhanced message for security intrusion
                                enhanced_message = f"{message_body} - {person_count} person(s) detected in ROI at {current_time_pass}"
                                # Save snapshot and notifications for security intrusion
                                save_snapshot_async(frame, cameraid, enhanced_message)
                                send_telegram_async(frame, enhanced_message, person_count, cameraname, current_time_pass)    
                                previous_snapshot_time = time.time()
                                logger.info(f"Security intrusion detected: {enhanced_message}")                           

                        cls_name = label_map[int(cls_id)]
                        
                        draw_bounding_box(frame, track_id, cls_name, confidence, x1, y1, x2, y2, color=(255, 0, 0), thickness=2)
                    
                else:
                    # No detections, just add basic frame info
                    cv2.putText(frame, 'No detections', (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                # Encode and stream frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ret:
                    logger.error("Failed to encode frame")
                    continue

                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
                # Control frame rate
                time.sleep(get_time_sleep())

    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")

    finally:
        cv2.destroyAllWindows()
        