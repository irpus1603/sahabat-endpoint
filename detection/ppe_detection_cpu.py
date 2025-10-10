
import cv2
import logging
import numpy as np
import time
from queue import Queue
from threading import Thread
from ultralytics import YOLO
from core.models import SystemConfig
from .send_notifications import send_telegram_notification
from .save_snapshots import save_snapshot_and_record
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global variable to cache the time_sleep value
_time_sleep = None

def get_time_sleep():
    """Get frame processing delay from config, with caching"""
    global _time_sleep
    if _time_sleep is None:
        try:
            _time_sleep = float(SystemConfig.get_value('frame_proc_delay', 0.020))
        except:
            _time_sleep = 0.020  # Default fallback
    return _time_sleep

class PPEDetection:
    """
    PPE (Personal Protective Equipment) Detection Class
    Handles detection of safety equipment like helmets and safety vests
    """
    
    def __init__(self, model_path="models/yolov8s_ppe.pt"):
        """Initialize PPE detection with YOLO model"""
        try:
            # Set environment variables to prevent fork crashes
            import os
            os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            
            self.det_model = YOLO(model_path, task='detect')
            self.label_map = self.det_model.names
            self.tracked_ids_with_missing_ppe = set()
            self.save_date_timestamp = {}
            whitelist=SystemConfig.get_value('ppe_det_cls_id_whitelist')
            try: 
                if whitelist:
                    self.required_safety_items = tuple(x.strip() for x in whitelist.split(','))
                else:
                    self.required_safety_items = ('Helmet', 'Safety-Vest')
            except Exception as e:
                self.required_safety_items = ('Helmet', 'Safety-Vest')

            logger.info(f"PPE Detection initialized with model: {model_path}")
            logger.info(f"Available classes: {list(self.label_map.values())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize PPE detection model: {e}")
            raise
    
    def is_blank_frame(self, frame, threshold=10):
        """Check if frame is blank/black"""
        if frame is None or frame.size == 0:
            return True
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stddev = np.std(gray_frame)
        return stddev < threshold

    def is_uniform_frame(self, frame, threshold=15):
        """Check if frame has uniform/no content"""
        if frame is None or frame.size == 0:
            return True
        return np.all(np.abs(frame - frame.mean()) < threshold)
    
    def is_point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon"""
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def get_polygon_from_coordinates(self, coordinates_str):
        """Convert coordinate string to polygon array"""
        if not coordinates_str:
            return None
        coords = list(map(int, coordinates_str.split(',')))
        return np.array(coords, dtype=np.int32).reshape((-1, 1, 2))

    def draw_blue_box(self, frame, track_id, cls_name, confidence, x1, y1, x2, y2, color_type='safe', thickness=1):
        """
        Draw blue box with consistent color reference and improved label visibility
        color_type: 'safe' (green), 'violation' (red), 'ppe' (blue), 'detection' (yellow)
        """
        # Color reference
        colors = {
            'safe': (0, 255, 0),        # Green for safe/compliant
            'violation': (0, 0, 255),   # Red for violations
            'ppe': (255, 0, 0),         # Blue for PPE equipment
            'detection': (0, 255, 255)  # Yellow for general detection
        }
        
        color = colors.get(color_type, (255, 0, 0))  # Default to blue
        
        # Draw rectangle box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Add label with improved visibility
        label = f"ID:{track_id} {cls_name} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        text_thickness = 1
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        
        # Add padding to background
        padding = 8
        bg_x1 = x1
        bg_y1 = y1 - text_height - padding * 2
        bg_x2 = x1 + text_width + padding * 2
        bg_y2 = y1
        
        # Ensure background doesn't go off screen
        if bg_y1 < 0:
            bg_y1 = y2
            bg_y2 = y2 + text_height + padding * 2
        
        # Choose background color based on status
        if color_type == 'violation' or 'Missing' in cls_name:
            # Light red background for missing PPE
            bg_color = (0, 0, 180)  # Light red (BGR format)
        else:
            # Semi-transparent black background for other cases
            bg_color = (0, 0, 0)
        
        # Draw background for better readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw text border (outline) for better visibility
        text_x = x1 + padding
        text_y = bg_y2 - padding if bg_y1 >= 0 else bg_y1 + text_height + padding
        
        # Draw text outline (black)
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, (0, 0, 0), text_thickness + 1, cv2.LINE_AA)
        
        # Draw main text (white)
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

    def calculate_bbox_overlap(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes"""
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])
        
        if x1_inter >= x2_inter or y1_inter >= y2_inter:
            return 0.0
        
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        ppe_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return intersection_area / ppe_area if ppe_area > 0 else 0.0
    
    def check_ppe_compliance(self, person_bbox, ppe_detections):
        """Check if a person is wearing required PPE items with improved overlap detection"""
        ppe_items = []
        overlap_threshold = 0.3  # Minimum 30% overlap required
        
        for ppe_detection in ppe_detections:
            ppe_bbox, ppe_cls_name = ppe_detection
            
            if ppe_cls_name in self.required_safety_items:
                # Calculate overlap between person and PPE bounding boxes
                overlap_ratio = self.calculate_bbox_overlap(person_bbox, ppe_bbox)
                
                # PPE must have significant overlap with person
                if overlap_ratio >= overlap_threshold:
                    ppe_items.append(ppe_cls_name)
        
        # Remove duplicates
        ppe_items = list(set(ppe_items))
        missing_items = [item for item in self.required_safety_items if item not in ppe_items]
        
        # Log detection details for debugging
        if len(ppe_items) > 0:
            logger.debug(f"PPE detected: {ppe_items} for person bbox {person_bbox}")
        if len(missing_items) > 0:
            logger.debug(f"Missing PPE: {missing_items} for person bbox {person_bbox}")
        
        return {
            'detected_items': ppe_items,
            'missing_items': missing_items,
            'is_compliant': len(missing_items) == 0
        }

    def draw_roi_polylines(self, frame, polygons, transparency=0.6):
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
        


    def process_ppe_detection(self, frame, polygon=None):
        """Process frame for PPE detection and return results"""
        if frame is None:
            return None
        
        # Skip blank or uniform frames
        if self.is_blank_frame(frame) or self.is_uniform_frame(frame):
            return None
            
        # Resize frame for processing
        original_height, original_width = frame.shape[:2]
        target_width = 640
        target_height = target_height = original_height * target_width // original_width
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        # Calculate scaling factors for polygon coordinates
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        
        # Scale polygon coordinates to match resized frame
        polygons = []
        if polygon is not None:
            scaled_polygon = polygon.copy()
            scaled_polygon[:, 0, 0] = (scaled_polygon[:, 0, 0] * scale_x).astype(np.int32)
            scaled_polygon[:, 0, 1] = (scaled_polygon[:, 0, 1] * scale_y).astype(np.int32)
            polygons = [scaled_polygon]
        
        # Note: ROI drawing is handled in process_ppe_video_stream to avoid duplication 

        # Run YOLO detection with higher confidence for more accurate results
        results = self.det_model.track(source=frame, persist=True, conf=0.3, iou=0.5, verbose=False)
        
        if not results or not results[0].boxes:
            return {'frame': frame, 'violations': 0, 'detections': []}

        # Process detections
        person_detections = []
        ppe_detections = []
        
        for track in results[0].boxes:
            bbox = list(map(int, track.xyxy[0]))
            cls_id = int(track.cls.item())
            cls_name = self.det_model.names[cls_id]
            confidence = track.conf.item()
            track_id = int(track.id.item()) if hasattr(track, 'id') and track.id is not None else -1
            
            if cls_name == "Person":
                person_detections.append({
                    'bbox': bbox,
                    'track_id': track_id,
                    'confidence': confidence
                })
            elif cls_name in self.required_safety_items:
                ppe_detections.append((bbox, cls_name))
                # Draw PPE detection with blue box
                x1, y1, x2, y2 = bbox
                self.draw_blue_box(frame, track_id, cls_name, confidence, x1, y1, x2, y2, 
                                 color_type='ppe', thickness=1)

        # Check PPE compliance for each person
        violations = 0
        detection_results = []
        
        for person in person_detections:
            person_bbox = person['bbox']
            track_id = person['track_id']
            confidence = person['confidence']
            
            ppe_status = self.check_ppe_compliance(person_bbox, ppe_detections)
            
            x1, y1, x2, y2 = person_bbox
            center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            if not ppe_status['is_compliant']:
                violations += 1
                missing_items = ppe_status['missing_items']
                
                # Draw violation box
                label_name = f"Missing({','.join(missing_items)})"
                self.draw_blue_box(frame, track_id, label_name, confidence, 
                                 x1, y1, x2, y2, color_type='violation', thickness=1)
                
                # Check if in ROI and handle notifications
                if len(polygons) == 0 or any(self.is_point_in_polygon(center_point, poly) for poly in polygons if poly is not None):
                    self.tracked_ids_with_missing_ppe.add(track_id)
                    
                    detection_results.append({
                        'track_id': track_id,
                        'bbox': person_bbox,
                        'missing_items': missing_items,
                        'in_roi': True,
                        'violation_type': 'ppe_missing'
                    })
            else:
                # Only draw safe box if person actually has PPE detected
                detected_items = ppe_status['detected_items']
                if len(detected_items) > 0:
                    label_name = f"Safe({','.join(detected_items)})"
                    self.draw_blue_box(frame, track_id, label_name, confidence, 
                                     x1, y1, x2, y2, color_type='safe', thickness=1)
                else:
                    # No PPE detected but required - this should be a violation
                    violations += 1
                    missing_items = list(self.required_safety_items)
                    label_name = f"Missing({','.join(missing_items)})"
                    self.draw_blue_box(frame, track_id, label_name, confidence, 
                                     x1, y1, x2, y2, color_type='violation', thickness=1)
                    
                    # Check if in ROI and handle notifications
                    if len(polygons) == 0 or any(self.is_point_in_polygon(center_point, poly) for poly in polygons if poly is not None):
                        self.tracked_ids_with_missing_ppe.add(track_id)
                        
                        detection_results.append({
                            'track_id': track_id,
                            'bbox': person_bbox,
                            'missing_items': missing_items,
                            'in_roi': True,
                            'violation_type': 'ppe_missing'
                        })

        return {
            'frame': frame,
            'violations': violations,
            'detections': detection_results,
            'total_people': len(person_detections)
        }


def save_snapshot_async(*args):
    """Async wrapper for saving snapshots"""
    Thread(target=save_snapshot_and_record, args=args).start()

def send_telegram_async(*args):
    """Async wrapper for sending telegram notifications"""
    Thread(target=send_telegram_notification, args=args).start()

# Lazy loading for PPE detector
ppe_detector = None

def get_ppe_detector():
    global ppe_detector
    if ppe_detector is None:
        ppe_detector = PPEDetection()
    return ppe_detector

# Queue for frames to be processed
frame_queue = Queue(maxsize=200)
list_class = (0,1,2,3,4,5,15,16)

# Global timestamp tracking for notifications (prevents spam)
save_date_timestamp = {}


# Function to capture video and push frames to the queue with robust HLS reconnection
def capture_video(hls_url, frame_queue):
    connection_attempts = 0
    max_attempts = 5
    base_retry_delay = 2  # Base delay in seconds
    consecutive_failures = 0
    last_frame_time = time.time()
    reconnect_threshold = 10  # Reconnect if no frames for 10 seconds
    
    while True:
        cap = None
        try:
            connection_attempts += 1
            logger.info(f"Attempting to connect to HLS source (attempt {connection_attempts}): {hls_url}")
            
            # Initialize the video capture optimized for HLS streams
            if hls_url[0:5].lower() == 'local':
                cap = cv2.VideoCapture(int(hls_url[6]))
            else:
                # HLS-optimized connection parameters
                cap = cv2.VideoCapture(hls_url, cv2.CAP_FFMPEG)
                
                # Set timeouts for HLS streams (longer than RTSP due to HLS nature)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)  # 15 second timeout for HLS
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)   # 8 second read timeout

            # HLS-optimized streaming settings
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Very small buffer for HLS (segments are already buffered)
            cap.set(cv2.CAP_PROP_FPS, 30)  # Set consistent FPS
            
            # HLS streams typically use H.264, but don't force it
            try:
                # Try to set preferred codec but don't fail if not supported
                
                if hasattr(cv2, 'VideoWriter_fourcc'):
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            except (AttributeError, Exception):
                logger.debug("H.264 codec setting not supported, using default")
                
            # Set reasonable frame dimensions for HLS streams
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # Check if the stream is open
            if not cap.isOpened():
                raise ConnectionError("Unable to open video stream")

            # Test if we can actually read a frame
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                raise ConnectionError("Unable to read initial frame from stream")

            logger.info(f"Video stream connected successfully on attempt {connection_attempts}")
            connection_attempts = 0  # Reset on successful connection
            consecutive_failures = 0
            last_frame_time = time.time()
            
            # Put the test frame in queue
            if not is_blank_frame(test_frame) and not is_uniform_frame(test_frame):
                frame_queue.put(test_frame)

            # Main frame capture loop with network monitoring
            frame_count = 0
            health_check_interval = 30  # Check connection health every 30 frames
            
            while True:
                current_time = time.time()
                
                # Check for connection timeout
                if current_time - last_frame_time > reconnect_threshold:
                    logger.warning(f"No frames received for {reconnect_threshold} seconds, forcing reconnection")
                    raise ConnectionError("Frame timeout - possible network interruption")
                
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    logger.warning(f"Frame read failed (consecutive failures: {consecutive_failures})")
                    
                    # Allow some tolerance for occasional failures
                    if consecutive_failures >= 5:
                        logger.error("Multiple consecutive frame read failures, reconnecting...")
                        raise ConnectionError("Multiple frame read failures")
                    
                    # Brief pause before retry
                    time.sleep(0.1)
                    continue
                
                # Reset failure counter on successful read
                consecutive_failures = 0
                last_frame_time = current_time
                frame_count += 1

                # Skip if the frame is blank or nearly uniform
                if is_blank_frame(frame) or is_uniform_frame(frame):
                    logger.debug("Skipping blank or low-content frame")
                    continue

                # Monitor queue size to prevent memory issues
                if frame_queue.qsize() > 50:
                    logger.warning("Frame queue is getting full, clearing old frames")
                    # Clear some frames to prevent memory buildup
                    try:
                        while frame_queue.qsize() > 30:
                            frame_queue.get_nowait()
                    except:
                        pass

                # Add frame to processing queue
                try:
                    frame_queue.put(frame, timeout=1)
                except:
                    logger.warning("Frame queue is full, skipping frame")
                    continue
                
                # Periodic connection health check
                if frame_count % health_check_interval == 0:
                    if not cap.isOpened():
                        logger.warning("Video capture object is no longer open, reconnecting")
                        raise ConnectionError("Video capture object closed")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"HLS capture error: {error_msg}")
            
            # Determine retry delay based on connection attempts and error type
            # HLS streams may need longer recovery times due to segment-based nature
            if "timeout" in error_msg.lower() or "network" in error_msg.lower():
                retry_delay = min(base_retry_delay * (2 ** min(connection_attempts, 4)), 45)  # Longer max for HLS
            else:
                retry_delay = base_retry_delay * min(connection_attempts, 3)
            
            # Exponential backoff with maximum attempts check
            if connection_attempts >= max_attempts:
                logger.error(f"Maximum connection attempts ({max_attempts}) reached. Waiting longer before retry...")
                retry_delay = 90  # Wait longer for HLS streams (90 seconds)
                connection_attempts = 0  # Reset for next cycle
            
            logger.info(f"Retrying HLS connection in {retry_delay} seconds...")
            
        finally:
            # Clean up resources
            if cap is not None:
                try:
                    cap.release()
                    logger.debug("Video capture released")
                except:
                    pass
                cap = None
            
            # Wait before reconnecting
            delay = locals().get('retry_delay', base_retry_delay)
            time.sleep(delay)

def create_robust_capture_with_monitoring(hls_url, frame_queue, status_callback=None, shared_status=None):
    """
    Enhanced wrapper for HLS capture_video with monitoring capabilities
    status_callback: function to call with connection status updates
    shared_status: shared status dict to check for stop signals
    """
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
            logger.info("Stop signal received in PPE capture thread, exiting...")
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
                cap = cv2.VideoCapture(int(hls_url[6]))  # Local camera
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
            if not is_blank_frame(test_frame) and not is_uniform_frame(test_frame):
                frame_queue.put(test_frame)
                total_frames += 1

            # Main streaming loop with enhanced monitoring
            frame_count = 0
            health_check_interval = 30
            
            while True:
                # Check for stop signal
                if connection_status.get('stop_requested', False):
                    logger.info("Stop signal received in PPE streaming loop, exiting...")
                    return
                    
                current_time = time.time()
                
                # Timeout check
                if current_time - last_frame_time > reconnect_threshold:
                    raise ConnectionError(f"Frame timeout after {reconnect_threshold} seconds")
                
                ret, frame = cap.read()
                
                # Check for stop signal immediately after frame read
                if connection_status.get('stop_requested', False):
                    logger.info("Stop signal received after PPE frame read, exiting...")
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

                # Skip bad frames
                if is_blank_frame(frame) or is_uniform_frame(frame):
                    continue

                # Queue management
                if frame_queue.qsize() > 50:
                    logger.warning("Clearing frame queue backlog")
                    try:
                        while frame_queue.qsize() > 30:
                            frame_queue.get_nowait()
                    except:
                        pass

                # Add frame to queue
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

def start_detection_stream(camera, hls_url):
    """
    Start a complete PPE detection stream with automatic thread management
    Returns a generator that yields detection frames and handles cleanup
    """
    # Create frame queue and connection status
    frame_queue = Queue(maxsize=50)
    connection_status = {'connected': False, 'error': None, 'stop_requested': False}
    
    def status_callback(status):
        connection_status.update(status)
        #logger.info(f"PPE Camera {camera.name} status: {status.get('status', 'unknown')}")
    
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
    
    time.sleep(1.0)
    
    # Check if capture started successfully
    if frame_queue.empty():
        time.sleep(3)
        if frame_queue.empty() and not connection_status.get('connected', False):
            error_msg = connection_status.get('error', 'Failed to start video capture')
            logger.error(f"PPE video capture failed to start for camera {camera.name}: {error_msg}")
            connection_status['stop_requested'] = True
            yield create_status_frame(f"Failed to start stream: {error_msg}")
            return
    
    logger.info(f"Starting PPE detection stream for camera {camera.name}")
    
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
            logger.info(f"Waiting for PPE capture thread to stop for camera {camera.name}...")
            capture_thread.join(timeout=3.0)
            if capture_thread.is_alive():
                logger.warning(f"PPE capture thread for camera {camera.name} did not stop gracefully within 3 seconds")
            else:
                logger.info(f"PPE capture thread for camera {camera.name} stopped successfully")
        
        raise
    except Exception as e:
        logger.error(f"PPE detection stream error for camera {camera.name}: {e}")
        connection_status['stop_requested'] = True
        yield create_status_frame(f"Stream error: {str(e)}")

def process_video_stream_with_reconnect(frame_queue, camera, connection_status):
    """
    Enhanced video stream processing with connection monitoring
    """
    try:
        # Use the original process_video_stream but with connection monitoring
        for frame_data in process_ppe_video_stream(frame_queue, camera):
            # Check connection status periodically
            if not connection_status.get('connected', True):
                # If disconnected, yield a status frame
                status_msg = connection_status.get('error', 'Connection lost')
                yield create_status_frame(f"Reconnecting... {status_msg}")
            else:
                yield frame_data
                
    except Exception as e:
        logger.error(f"Stream processing error: {e}")
        yield create_status_frame(f"Stream error: {str(e)}")

def process_ppe_video_stream(frame_queue, camera):
    """Process PPE detection video stream"""
    cameraid = camera.id
    cameraname = camera.name
    
    # Initialize PPE detection
    ppe_detector = PPEDetection()
    
    # Get ROI coordinates
    roi_coords = camera.get_roi_coordinates()
    polygons = []
    
    if roi_coords is not None:
        if len(roi_coords.shape) == 2 and roi_coords.shape[1] == 2:
            polygons = [roi_coords.reshape(-1, 1, 2)]
        else:
            polygons = [roi_coords] if len(roi_coords.shape) == 3 else [roi_coords.reshape(-1, 1, 2)]
    else:
        polygons = []
        logger.warning(f"PPE Camera {cameraname} - No ROI coordinates found")
    
    previous_snapshot_time = 0.0

    try:
        while True:
            frame = frame_queue.get()  # Get frame from queue
            if frame is None:
                logger.error("Cannot capture frame")
                continue

            current_time_pass = datetime.now().strftime('%d-%m-%y %H:%M:%S')
            message_body = f"PPE violation detected on camera {cameraname} at {current_time_pass}"

            # Resize frame for processing
            original_height, original_width = frame.shape[:2]
            target_width = 640
            target_height = 480
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            # Calculate scaling factors for polygon coordinates
            scale_x = target_width / original_width
            scale_y = target_height / original_height

            # Scale polygons to match resized frame for both display and detection
            scaled_polygons = []
            if len(polygons) > 0:
                for polygon in polygons:
                    if polygon is not None:
                        # Scale polygon coordinates to match resized frame
                        scaled_polygon = polygon.copy()
                        scaled_polygon[:, 0, 0] = (scaled_polygon[:, 0, 0] * scale_x).astype(np.int32)
                        scaled_polygon[:, 0, 1] = (scaled_polygon[:, 0, 1] * scale_y).astype(np.int32)
                        scaled_polygons.append(scaled_polygon)
                
                # Draw ROI with transparent gray fill
                ppe_detector.draw_roi_polylines(frame, scaled_polygons, 0.3)

            # Process PPE detection with properly scaled polygon
            detection_result = ppe_detector.process_ppe_detection(frame, scaled_polygons[0] if scaled_polygons else None)
            
            if detection_result is not None and isinstance(detection_result, dict) and 'frame' in detection_result:
                # Extract frame from detection result dictionary
                processed_frame = detection_result['frame']
                
                # Handle PPE violations and snapshots
                if detection_result.get('violations', 0) > 0 and len(detection_result.get('detections', [])) > 0:
                    # Check if violations are within ROI
                    violations_in_roi = [d for d in detection_result['detections'] if d.get('in_roi', False)]
                    
                    if violations_in_roi:
                        current_time = time.time()
                        
                        # Get snapshot timer from config (same as object_detection.py)
                        snapshot_timer_value = SystemConfig.get_value('snapshot_counter')
                        snapshot_timer = int(snapshot_timer_value) if snapshot_timer_value else 60
                        
                        if current_time - previous_snapshot_time >= snapshot_timer:
                            # Enhanced message for PPE violations
                            violation_count = len(violations_in_roi)
                            enhanced_message = f"{message_body} - {violation_count} PPE violation(s) detected in ROI at {current_time_pass}"
                            
                            # Save snapshot and send notifications asynchronously
                            save_snapshot_async(processed_frame, cameraid, enhanced_message)
                            send_telegram_async(processed_frame, enhanced_message, violation_count, cameraname, current_time_pass)
                            
                            previous_snapshot_time = time.time()
                            logger.info(f"PPE violation detected and recorded: {enhanced_message}")
            else:
                # Use original frame if detection returned None or invalid result
                processed_frame = frame
            
            # Validate frame before encoding
            if processed_frame is not None and isinstance(processed_frame, np.ndarray):
                # Encode frame for streaming
                ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    frame_bytes = buffer.tobytes()
                    try:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        #time.sleep(0.015)  # 30 FPS
                    except (GeneratorExit, ConnectionError, BrokenPipeError):
                        logger.info(f"PPE streaming client disconnected for camera {cameraname}")
                        break
            else:
                logger.warning(f"Invalid frame type for encoding: {type(processed_frame)}")
                       
    except Exception as e:
        logger.error(f"PPE video stream processing error: {e}")
        yield create_status_frame(f"PPE processing error: {str(e)}")

def create_status_frame(message):
    """Create a simple status frame with text message"""
    
    
    # Create a black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add status text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 255)  # Yellow
    thickness = 1
    
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

