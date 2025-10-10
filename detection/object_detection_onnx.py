
import cv2
import logging
import numpy as np
import onnxruntime as ort
from threading import Thread
from core.models import SystemConfig
import time
from queue import Queue
from .send_notifications import send_telegram_notification
from .save_snapshots import save_snapshot_and_record
from datetime import datetime


# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Removed unused get_time_sleep function

# Lazy loading for ONNX model
det_model = None
label_map = None


def detect_device():
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        device = 'cuda'
        logger.info("GPU detected: Using CUDA execution provider")
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        device = 'cpu'
        logger.info("Using CPU execution provider")
        return ['CPUExecutionProvider']

def get_model():
    global det_model, label_map
    providers = detect_device()
    if det_model is None:
        try:
            det_model = ort.InferenceSession("models/yolov8n.onnx", providers=providers)
            logger.info(f"ONNX model loaded successfully with providers: {providers}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
        # COCO class names for YOLOv8
        label_map = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
            6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
            11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
            16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
            27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
            46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
            51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
            56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
            61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
            67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
            72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
            77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }
    return det_model, label_map

# Configuration
try:
    whitelist = SystemConfig.get_value('obj_det_cls_id_whitelist')
    list_class = [int(x.strip()) for x in whitelist.split(',')] if whitelist else [0,1,2,3]
except Exception as e:
    # Table doesn't exist yet (during migrations) or other error
    list_class = [0,1,2,3]

def preprocess_image(image, input_size=(640, 640)):
    """Preprocess image for ONNX model inference"""
    # Resize image
    resized = cv2.resize(image, input_size)
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    # Change format from HWC to CHW
    transposed = np.transpose(normalized, (2, 0, 1))
    # Add batch dimension
    batched = np.expand_dims(transposed, axis=0)
    return batched

def postprocess_outputs(outputs, conf_threshold=0.3, iou_threshold=0.5, input_size=(640, 640), original_size=None):
    """Simplified post-processing of ONNX model outputs"""
    predictions = outputs[0]
    
    # Handle YOLOv8 format: transpose if needed
    if len(predictions.shape) == 3 and predictions.shape[1] == 84:
        predictions = np.transpose(predictions, (0, 2, 1))
    
    detections = []
    pred = predictions[0]  # Only process first batch
    
    # Extract boxes and scores
    boxes = pred[:, :4]  # x_center, y_center, width, height
    scores = pred[:, 4:]  # class scores
    
    # Find valid detections
    max_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)
    valid_mask = max_scores > conf_threshold
    
    if not np.any(valid_mask):
        return detections
    
    # Get valid detections
    valid_boxes = boxes[valid_mask]
    valid_scores = max_scores[valid_mask]
    valid_class_ids = class_ids[valid_mask]
    
    # Convert to corner format and scale
    x_center, y_center, width, height = valid_boxes.T
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    # Scale to original size if provided
    if original_size:
        orig_h, orig_w = original_size
        input_w, input_h = input_size
        x1 = x1 * orig_w / input_w
        x2 = x2 * orig_w / input_w
        y1 = y1 * orig_h / input_h
        y2 = y2 * orig_h / input_h
    
    # Simple NMS using OpenCV
    try:
        boxes_list = [[x1[i], y1[i], x2[i], y2[i]] for i in range(len(x1))]
        indices = cv2.dnn.NMSBoxes(boxes_list, valid_scores.tolist(), conf_threshold, iou_threshold)
        
        if len(indices) > 0:
            for idx in indices.flatten():
                detections.append({
                    'bbox': [int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx])],
                    'confidence': float(valid_scores[idx]),
                    'class_id': int(valid_class_ids[idx])
                })
    except Exception as e:
        # Fallback: return all detections without NMS
        logger.warning(f"NMS failed, returning all detections: {e}")
        for i in range(len(valid_scores)):
            detections.append({
                'bbox': [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
                'confidence': float(valid_scores[i]),
                'class_id': int(valid_class_ids[i])
            })
    
    return detections

# Cache input details for better performance
_input_name = None
_input_size = None

def run_onnx_inference(model, frame):
    """Run inference on ONNX model"""
    global _input_name, _input_size
    
    try:
        # Cache input details on first run
        if _input_name is None:
            _input_name = model.get_inputs()[0].name
            input_shape = model.get_inputs()[0].shape
            _input_size = (input_shape[2], input_shape[3]) if len(input_shape) == 4 else (640, 640)
        
        # Preprocess and run inference
        preprocessed = preprocess_image(frame, _input_size)
        outputs = model.run(None, {_input_name: preprocessed})
        
        # Post-process outputs 
        detections = postprocess_outputs(outputs, conf_threshold=0.3, original_size=frame.shape[:2])
        
        return detections
    except Exception as e:
        logger.error(f"ONNX inference error: {e}")
        return []

def is_bad_frame(frame):
    """Simple check if frame is blank or has no content"""
    if frame is None or frame.size == 0:
        return True
    
    # Quick check using small sample region
    h, w = frame.shape[:2]
    sample = frame[h//2-50:h//2+50, w//2-50:w//2+50]  # 100x100 center sample
    
    if sample.size == 0:
        return True
        
    # Check if frame has sufficient variance (not blank/uniform)
    return np.var(sample) < 10

# Removed unused calculate_frame_difference function

def test_video_source_connectivity(hls_url, timeout=2):
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
         print(f"detected url: {hls_url}")
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
         if ret and frame is not None and not is_bad_frame(frame):
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

# Simplified - removed wrapper function, use capture_video_with_monitoring directly

def capture_video_with_monitoring(hls_url, frame_queue, status_update_func, connection_status):
    """Enhanced HLS capture function with status monitoring"""
    connection_attempts = 0
    max_attempts = 5
    base_retry_delay = 0.5  # Slightly longer for HLS
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
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)  # Longer for HLS
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)   # Longer for HLS

            # HLS-optimized streaming settings
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Smaller for HLS
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
            if not is_bad_frame(test_frame):
                frame_queue.put(test_frame)
                total_frames += 1

            # Main streaming loop with enhanced monitoring
            frame_count = 0
            health_check_interval = 150
            
            while True:
                # Check for stop signal
                if connection_status.get('stop_requested', False):
                    logger.info("Stop signal received in streaming loop, exiting...")
                    return
                    
                current_time = time.time()
                
                # Timeout check
                if current_time - last_frame_time > reconnect_threshold:
                    raise ConnectionError(f"Frame timeout after {reconnect_threshold} seconds")
                
                ret, frame = cap.read()
                
                # Check for stop signal immediately after frame read
                if connection_status.get('stop_requested', False):
                    logger.info("Stop signal received after frame read, exiting...")
                    return
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    logger.warning(f"Frame read failed (consecutive: {consecutive_failures})")
                    
                    if consecutive_failures >= 5:
                        raise ConnectionError("Multiple consecutive frame failures")
                    
                    
                    time.sleep(0.1)
                    continue
                
                # Success - reset counters and update status
                consecutive_failures = 0
                last_frame_time = current_time
                frame_count += 1
                total_frames += 1

                # Skip bad frames - simplified check
                if is_bad_frame(frame):
                    continue

                # Queue management exactly like PPE detection
                if frame_queue.qsize() > 50:
                    logger.warning("Clearing frame queue backlog")
                    try:
                        while frame_queue.qsize() > 30:
                            frame_queue.get_nowait()
                    except:
                        pass

                # Add frame to queue exactly like PPE detection
                try:
                    frame_queue.put(frame, timeout=1)
                except:
                    logger.warning("Frame queue full, skipping frame")
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
            logger.error(f"Video capture error: {error_msg}")
            
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
                logger.error(f"Max attempts reached, waiting 60 seconds...")
                retry_delay = 60
                connection_attempts = 0
            
            logger.info(f"Retrying in {retry_delay} seconds...")
            
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

# Async helpers
def save_snapshot_async(*args):
    Thread(target=save_snapshot_and_record, args=args).start()

def send_telegram_async(*args):
    Thread(target=send_telegram_notification, args=args).start()

def start_detection_stream(camera, hls_url):
   
    # Create frame queue and connection status
    frame_queue = Queue(maxsize=50)
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
    
    # Start simplified capture thread
    def update_status(status_dict):
        connection_status.update(status_dict)
        if status_callback:
            try:
                status_callback(connection_status.copy())
            except Exception as e:
                logger.error(f"Status callback error: {e}")
    
    capture_thread = Thread(
        target=capture_video_with_monitoring,
        args=(hls_url, frame_queue, update_status, connection_status),
        daemon=True
    )
    capture_thread.start()
    
    # Wait for capture to start
    time.sleep(1.0)
    
    # Check if capture started successfully
    if frame_queue.empty():
        #time.sleep(get_time_sleep())
        if frame_queue.empty() and not connection_status.get('connected', False):
            error_msg = connection_status.get('error', 'Failed to start video capture')
            logger.error(f"Video capture failed to start for camera {camera.name}: {error_msg}")
            connection_status['stop_requested'] = True
            yield create_status_frame(f"Failed to start stream: {error_msg}")
            return
    
    logger.info(f"Starting detection stream for camera {camera.name}")
    
    try:
        # Direct stream processing with connection monitoring
        for frame_data in process_video_stream(frame_queue, camera):
            # Check connection status
            if not connection_status.get('connected', True):
                status_msg = connection_status.get('error', 'Connection lost')
                yield create_status_frame(f"Reconnecting... {status_msg}")
            else:
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

# Removed wrapper function - integrated into main stream processing

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
    """Draw ROI polygons with transparency - simplified"""
    if not polygons:
        return
    
    try:
        overlay = frame.copy()
        for polygon in polygons:
            if polygon is not None and len(polygon) > 2:
                # Ensure proper format
                if len(polygon.shape) == 3:
                    points = polygon.reshape(-1, 2)
                else:
                    points = polygon
                
                points = np.array(points, dtype=np.int32)
                # Clip to frame bounds
                points[:, 0] = np.clip(points[:, 0], 0, frame.shape[1] - 1)
                points[:, 1] = np.clip(points[:, 1], 0, frame.shape[0] - 1)
                
                # Draw filled polygon and border
                polygon_cv = points.reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [polygon_cv], (255, 0, 0))  # Blue fill
                cv2.polylines(overlay, [polygon_cv], True, (255, 0, 0), 2)  # Blue border
        
        # Blend with transparency
        cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0, frame)
    except Exception as e:
        logger.error(f"Error drawing ROI: {e}")

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
    
    # Load model once at initialization
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
    
    # Direct inference like PPE detection - no async complexity
    last_detections = []  # Cache last detections for smooth display

    try:
        while True:
                # Simple frame retrieval like PPE detection
                frame = frame_queue.get()  # Simple blocking call, no timeout
                    
                if frame is None:
                    logger.error(f"Camera {cameraname} - Cannot capture frame (frame is None)")
                    continue
                
                # No timing analysis - keep it simple like PPE detection

                current_time_pass = datetime.now().strftime('%d-%m-%y %H:%M:%S')
                message_body = f"Security intrusion detected on camera {cameraname} at {current_time_pass}"
                
                # Simple frame resizing
                original_height, original_width = frame.shape[:2]
                target_width = 640
                target_height = target_height = original_height * target_width // original_width
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                
                # Calculate scaling factors for polygon coordinates
                scale_x = target_width / original_width
                scale_y = target_height / original_height
                #logger.info(f"Camera {cameraname} - Resized to: {target_width}x{target_height}, scale factors: x={scale_x:.3f}, y={scale_y:.3f}")

                #print(f"polygons: {polygons}")
                # Draw all polygons on the video with proper scaling using transparent overlay
                if len(polygons) > 0:
                    scaled_polygons = []
                    for i, polygon in enumerate(polygons):
                        if polygon is not None and polygon.size > 0:
                            try:
                                # Scale polygon coordinates to match resized frame
                                scaled_polygon = polygon.copy()
                                
                                # Handle different polygon formats
                                if len(scaled_polygon.shape) == 3:
                                    # Format: (N, 1, 2)
                                    scaled_polygon[:, 0, 0] = (scaled_polygon[:, 0, 0] * scale_x).astype(np.int32)
                                    scaled_polygon[:, 0, 1] = (scaled_polygon[:, 0, 1] * scale_y).astype(np.int32)
                                elif len(scaled_polygon.shape) == 2:
                                    # Format: (N, 2)
                                    scaled_polygon[:, 0] = (scaled_polygon[:, 0] * scale_x).astype(np.int32)
                                    scaled_polygon[:, 1] = (scaled_polygon[:, 1] * scale_y).astype(np.int32)
                                    # Reshape to OpenCV format (N, 1, 2)
                                    scaled_polygon = scaled_polygon.reshape(-1, 1, 2)
                                
                                scaled_polygons.append(scaled_polygon)
                            except Exception as e:
                                logger.error(f"Camera {cameraname} - Error scaling polygon {i}: {e}")
                                continue
                    
                    # Draw ROI with transparent blue fill
                    draw_roi_polylines(frame, scaled_polygons, transparency=0.1)
                

                # Use detections from synchronous inference above
                if 'detections' not in locals():
                    detections = last_detections
                
                # Direct synchronous inference like PPE detection
                try:
                    detections = run_onnx_inference(model, frame.copy())
                    if detections:
                        last_detections = detections
                except Exception as e:
                    logger.warning(f"Camera {cameraname} - Inference failed: {e}")
                    detections = last_detections  # Use cached detections

                # Check if detections contain results
                if detections and len(detections) > 0:
                    person_count = sum(1 for det in detections if det['class_id'] == 0)
                    # Use the new function to draw detection counter
                    draw_detection_counter(frame, person_count, "people")

                    # Only process detections in whitelist
                    for i, detection in enumerate(detections):
                        cls_id = detection['class_id']
                        if cls_id not in list_class:
                            continue
                            
                        x1, y1, x2, y2 = detection['bbox']
                        confidence = detection['confidence']
                        track_id = i + 1  # Simple ID assignment since we don't have tracking
                        cls_name = label_map[cls_id]
                        
                        # Draw bounding box
                        draw_bounding_box(frame, track_id, cls_name, confidence, x1, y1, x2, y2, color=(255, 0, 0), thickness=2)
                        
                        # Only check ROI for person detections (class_id 0) to optimize performance
                        if cls_id == 0 and len(polygons) > 0:
                            center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                            
                            # Check if center point is inside any of the ROI polygons
                            for polygon in polygons:
                                if polygon is not None:
                                    # Scale polygon coordinates to match resized frame for detection
                                    scaled_polygon = polygon.copy()
                                    scaled_polygon[:, 0, 0] = (scaled_polygon[:, 0, 0] * scale_x).astype(np.int32)
                                    scaled_polygon[:, 0, 1] = (scaled_polygon[:, 0, 1] * scale_y).astype(np.int32)
                                    
                                    if is_point_in_polygon(center_point, scaled_polygon):
                                        current_time = time.time()
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
                                        break  # Exit polygon loop once found
                    
                else:
                    #logger.info(f"No detections for camera {cameraname} at {current_time_pass}")
                    # No detections, just add basic frame info
                    draw_detection_counter(frame, 0, "people")

                # Simple frame encoding like PPE detection
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ret:
                    logger.error(f"Camera {cameraname} - Failed to encode frame.")
                    continue

                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                # No artificial sleep like PPE detection - natural frame rate
                

    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")

    finally:
        # Clean up resources - no async inference to clean up
        logger.info(f"Camera {cameraname} stream processing ended")
        