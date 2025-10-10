
import logging
from threading import Thread

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global timestamp tracking for notifications (prevents spam)
save_date_timestamp = {}

# Global database manager instance (lazy-loaded)
_database_manager = None

def get_database_manager():
    """Get database manager with lazy loading"""
    global _database_manager
    if _database_manager is None:
        from .database_operations import DatabaseManager
        _database_manager = DatabaseManager()
    return _database_manager

def save_snapshot_with_intrusion_async(frame, cameraid, message_body, detection_type, object_count, intrusion_details=None):
    """Async wrapper for intrusion detection saving"""
    Thread(target=get_database_manager().save_intrusion_detection, 
           args=(frame, cameraid, message_body, detection_type, object_count, intrusion_details)).start()

def send_telegram_async(*args):
    from detection.send_notifications import send_telegram_notification
    Thread(target=send_telegram_notification, args=args).start()

def start_detection_stream(camera, hls_url):
    """Start detection stream using new modular architecture"""
    
    # Import modules only when function is called
    from .video_capture import VideoStreamHandler
    from .processor import StreamProcessor
    from .utils import create_status_frame
    
    # Create video stream handler
    video_handler = VideoStreamHandler(camera)
    
    # Start video stream
    capture_thread = video_handler.start_stream(hls_url)
    if capture_thread is None:
        yield create_status_frame(f"Failed to start stream")
        return
    
    logger.info(f"Starting detection stream for camera {camera.name}")
    
    try:
        # Process video stream with new processor
        stream_processor = StreamProcessor(camera)
        for frame_data in stream_processor.process_video_stream(video_handler.frame_queue):
            yield frame_data
            
    except GeneratorExit:
        # Client disconnected - cleanup resources
        logger.info(f"Client disconnected for camera {camera.name}, cleaning up...")
        video_handler.stop_stream()
        
        # Wait for capture thread to stop
        if capture_thread and capture_thread.is_alive():
            logger.info(f"Waiting for capture thread to stop for camera {camera.name}...")
            capture_thread.join(timeout=3.0)
            if capture_thread.is_alive():
                logger.warning(f"Capture thread for camera {camera.name} did not stop gracefully within 3 seconds")
            else:
                logger.info(f"Capture thread for camera {camera.name} stopped successfully")
        
        raise
    except Exception as e:
        logger.error(f"Detection stream error for camera {camera.name}: {e}")
        video_handler.stop_stream()
        yield create_status_frame(f"Stream error: {str(e)}")












        