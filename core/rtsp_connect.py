from hashlib import new
import cv2
import time
from .models import Camera
import logging
import numpy as np
import concurrent.futures
from django.core.cache import cache

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class RTSPConnectionManager:
    """Optimized class-based RTSP connection manager for SOCA cameras"""
    
    def __init__(self, cache_timeout=300):
        self.cache_timeout = cache_timeout
        self.logger = logging.getLogger(__name__)
    
    def get_camera_url(self, camera):
        """Get RTSP URL for SOCA Camera model or USB camera device"""
        # For SOCA Camera model, use rtsp_url field directly
        rtsp_url = camera.rtsp_url
        camera_source = camera.camera_source.lower()
        
        if not rtsp_url:
            self.logger.error(f"No RTSP URL configured for camera {camera.id}")
            return None
        
        # Handle USB/local camera devices
        if camera_source == 'local':
            try:
                device_id = int(rtsp_url.split(':')[1])
                self.logger.info(f"Camera {camera.id} configured for USB device: {device_id}")
                return device_id  # Return device ID for USB cameras
            except (ValueError, IndexError):
                self.logger.error(f"Invalid local camera format: {rtsp_url}. Use format 'local:0', 'local:1', etc.")
                return None
        
        elif camera_source == 'video_file':
            # Handle relative paths for video files
            if rtsp_url.startswith('/media/'):
                rtsp_url = rtsp_url[1:]  # Remove leading slash to make it relative
            self.logger.info(f"Camera {camera.id} configured for video file: {rtsp_url}")
            return rtsp_url  # Return file path for video files
        elif camera_source == 'youtube':
            self.logger.info(f"Camera {camera.id} configured for YouTube URL: {rtsp_url}")
            return rtsp_url  # Return YouTube URL directly
        elif camera_source == 'rtsp':
            if not rtsp_url.startswith('rtsp://'):
                self.logger.error(f"Invalid RTSP URL for camera {camera.id}: {rtsp_url}")
                return None
            else:
                # Add authentication if provided
                if camera.username and camera.password:
                    # Remove any existing rtsp:// prefix and credentials
                    clean_url = rtsp_url.replace('rtsp://', '')
                    # Remove any existing credentials from the URL
                    if '@' in clean_url:
                        clean_url = clean_url.split('@', 1)[1]
                    # Construct clean URL with proper credentials
                    rtsp_url = f"rtsp://{camera.username}:{camera.password}@{clean_url}"
                
                self.logger.info(f"Camera {camera.id} RTSP URL: {rtsp_url}")
                return camera, rtsp_url
    
    def get_camera_by_id(self, camera_id):
        """Get camera with caching to reduce database hits"""
        cache_key = f"camera_{camera_id}"
        camera = cache.get(cache_key)
        
        if camera is None:
            try:
                camera = Camera.objects.get(id=camera_id, is_active=True)
                cache.set(cache_key, camera, self.cache_timeout)
                self.logger.info(f"Camera {camera_id} loaded from database")
            except Camera.DoesNotExist:
                self.logger.error(f"Camera with ID {camera_id} does not exist or is inactive")
                return None
        else:
            self.logger.debug(f"Camera {camera_id} loaded from cache")
        
        return camera
    
    def test_camera_connection(self, camera):
        """Test camera connection with timeout and retry logic"""
        camera_source = self.get_camera_url(camera)
        if camera_source is None:
            return False
        
        try:
            # Handle USB cameras differently
            if isinstance(camera_source, int):
                # USB camera - use device ID directly
                self.logger.info(f"Testing USB camera {camera.id} with device ID: {camera_source}")
                cap = cv2.VideoCapture(camera_source)
                if not cap.isOpened():
                    self.logger.warning(f"Cannot open USB camera {camera.id} on device {camera_source}")
                    cap.release()
                    return False
            else:
                # RTSP camera - try FFMPEG backend first
                cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    self.logger.warning(f"FFMPEG backend failed for camera {camera.id}, trying default backend")
                    cap.release()
                    cap = cv2.VideoCapture(camera_source)
            
            if not cap.isOpened():
                self.logger.warning(f"Cannot open camera {camera.id} stream")
                cap.release()
                return False
            
            # Try to read one frame to verify connection
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                self.logger.info(f"Camera {camera.id} connection successful")
                return True
            else:
                self.logger.warning(f"Camera {camera.id} failed to read frame")
                return False
                
        except Exception as e:
            self.logger.error(f"Error testing camera {camera.id}: {str(e)}")
            return False
    
    def generate_frames(self, camera_id, frame_rate=15, target_width=1280):
        """Optimized frame generation with better error handling and caching"""
        
        # Create text-based fallback frame
        no_video_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        no_video_frame[:] = (45, 45, 45)  # Dark gray background
        
        # Add border
        cv2.rectangle(no_video_frame, (5, 5), (315, 235), (80, 80, 80), 2)
        
        # Add main text
        cv2.putText(no_video_frame, 'NO VIDEO', (85, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(no_video_frame, 'SIGNAL', (105, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add status indicator
        cv2.circle(no_video_frame, (160, 180), 8, (0, 0, 255), -1)  # Red circle
        cv2.putText(no_video_frame, 'OFFLINE', (120, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Get camera from cache/database
        camera = self.get_camera_by_id(camera_id)
        if not camera:
            self.logger.error(f"Camera {camera_id} not found")
            yield from self._yield_fallback_frames(no_video_frame, frame_rate)
            return
        
        # Get camera source (RTSP URL or USB device ID)
        camera_source = self.get_camera_url(camera)
        if camera_source is None:
            self.logger.error(f"No valid camera source for camera {camera_id}")
            yield from self._yield_fallback_frames(no_video_frame, frame_rate)
            return
        
        consecutive_failures = 0
        max_failures = 10
        cap = None
        is_usb_camera = isinstance(camera_source, int)
        
        try:
            while True:
                if is_usb_camera:
                    # USB camera - use device ID directly
                    self.logger.info(f"Opening USB camera {camera_id} with device ID: {camera_source}")
                    cap = cv2.VideoCapture(camera_source)
                else:
                    # RTSP/video file camera - use FFMPEG backend with fallback
                    self.logger.info(f"Opening camera {camera_id} with URL: {camera_source}")
                    cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        self.logger.warning(f"FFMPEG backend failed for {camera_source}, trying default backend")
                        cap.release()
                        cap = cv2.VideoCapture(camera_source)
                
                # Optimize capture settings
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                cap.set(cv2.CAP_PROP_FPS, frame_rate)
                
                # Additional settings for USB cameras
                if is_usb_camera:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(target_width * 0.75))  # 4:3 aspect ratio
                
                if not cap.isOpened():
                    self.logger.error(f"Cannot open camera {camera_id} stream. Retrying...")
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        yield from self._yield_fallback_frames(no_video_frame, frame_rate)
                        return
                    time.sleep(3)
                    continue
                
                self.logger.info(f"Camera {camera_id} stream opened successfully")
                consecutive_failures = 0  # Reset on successful connection
                
                while True:
                    success, frame = cap.read()
                    if not success:
                        # Check if this is a video file that reached the end
                        if isinstance(camera_source, str) and not camera_source.startswith('rtsp://'):
                            # For video files, restart from beginning when reaching end
                            self.logger.info(f"Video file {camera_id} reached end, restarting from beginning")
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            success, frame = cap.read()
                            if success:
                                consecutive_failures = 0
                            else:
                                consecutive_failures += 1
                        else:
                            self.logger.warning(f"Failed to read frame from camera {camera_id}")
                            consecutive_failures += 1
                        
                        if consecutive_failures >= max_failures:
                            break
                        continue
                    
                    # Skip blank or uniform frames
                    if self._is_blank_frame(frame) or self._is_uniform_frame(frame):
                        continue
                    
                    # Resize frame if needed
                    if frame.shape[1] != target_width:
                        scale_ratio = target_width / frame.shape[1]
                        new_height = int(frame.shape[0] * scale_ratio)
                        frame = cv2.resize(frame, (target_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # Encode and yield frame
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                
                cap.release()
                
                # If we reach here, there were too many failures
                if consecutive_failures >= max_failures:
                    self.logger.error(f"Max failures reached for camera {camera_id}. Switching to fallback.")
                    yield from self._yield_fallback_frames(no_video_frame, frame_rate)
                    return
                
        except Exception as e:
            self.logger.exception(f"Error in frame generation for camera {camera_id}: {e}")
            yield from self._yield_fallback_frames(no_video_frame, frame_rate)
        finally:
            try:
                if cap is not None:
                    cap.release()
            except:
                pass
    
    def _yield_fallback_frames(self, no_video_frame, frame_rate):
        """Yield fallback frames when camera is unavailable"""
        while True:
            ret, buffer = cv2.imencode('.jpg', no_video_frame)
            if not ret:
                self.logger.error("Failed to encode fallback frame")
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1 / frame_rate)
    
    def _is_blank_frame(self, frame, threshold=10):
        """Check if frame is blank/black"""
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            stddev = np.std(gray_frame)
            return stddev < threshold
        except:
            return True
    
    def _is_uniform_frame(self, frame, threshold=15):
        """Check if frame has uniform color"""
        try:
            return np.all(np.abs(frame - frame.mean()) < threshold)
        except:
            return True

    def get_all_cameras_status(self):
        """Get status of all cameras with minimal database queries"""
        cache_key = "all_cameras_status"
        cameras_status = cache.get(cache_key)
        
        if cameras_status is None:
            cameras = Camera.objects.all().values('id', 'name', 'rtsp_url', 'is_active', 'username', 'password')
            cameras_status = []
            
            # Test connections in parallel for better performance
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_camera = {}
                
                for camera_data in cameras:
                    if camera_data['is_active']:
                        # Create camera object for testing
                        camera_obj = type('Camera', (), camera_data)
                        future = executor.submit(self.test_camera_connection, camera_obj)
                        future_to_camera[future] = camera_data
                
                for future in concurrent.futures.as_completed(future_to_camera):
                    camera_data = future_to_camera[future]
                    try:
                        connection_status = future.result(timeout=10)
                        camera_data['connection_status'] = connection_status
                    except Exception as exc:
                        self.logger.error(f"Camera {camera_data['id']} test failed: {exc}")
                        camera_data['connection_status'] = False
                    
                    cameras_status.append(camera_data)
            
            # Cache for 60 seconds to avoid frequent testing
            cache.set(cache_key, cameras_status, 60)
            
        return cameras_status
    
    def capture_single_frame(self, camera):
        """Capture a single frame from camera and return as base64 encoded image"""
        import base64

        #camera_source = self.get_camera_url(camera)
        camera_source = camera.camera_source.lower()
        camera_id = camera.id
        print("Camera Source:", camera_source)

        if camera_source is None:
            return None
        
        try:
            # Add specific check for video files
            if camera_source == 'video_file':
                import os
                file_path = camera.rtsp_url
                if file_path.startswith('/media/'):
                    file_path = file_path[1:]  # Remove leading slash
                if not os.path.exists(file_path):
                    self.logger.error(f"Video file not found: {file_path}")
                    return None

            if camera_source == 'local':
                try:
                    url = int(camera.rtsp_url.split(':')[1])
                    cap = cv2.VideoCapture(url)
                except (ValueError, IndexError):
                    self.logger.error(f"Invalid local camera format: {camera.rtsp_url}. Use format 'local:0', 'local:1', etc.")
                    return None

            elif camera_source == 'video_file':
                url = camera.rtsp_url
                # Handle relative paths for video files
                if url.startswith('/media/'):
                    url = url[1:]  # Remove leading slash to make it relative
                
                # Try FFMPEG backend first, then fall back to default
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    self.logger.warning(f"FFMPEG backend failed for video file {url}, trying default backend")
                    cap.release()
                    cap = cv2.VideoCapture(url)

            elif camera_source == 'youtube':
                url = camera.rtsp_url
                cap = cv2.VideoCapture(url)
            
            elif camera_source == 'rtsp':
                url = camera.rtsp_url
                if camera.username and camera.password:
                    clean_url = url.replace('rtsp://', '')
                    if '@' in clean_url:
                        clean_url = clean_url.split('@', 1)[1]
                    url = f"rtsp://{camera.username}:{camera.password}@{clean_url}"
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            else:
                self.logger.error(f"Unsupported camera source type for camera {camera.id}")
                return None 
            
            if not cap.isOpened():
                self.logger.warning(f"Cannot open camera {camera.id} for frame capture")
                return None
            
            # Try to read one frame with better handling for video files
            if camera_source == 'video_file':
                # For video files, seek to a frame that's not at the beginning
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 30:
                    # Seek to frame 30 to avoid blank frames at the beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
                time.sleep(0.5)  # Small delay to ensure seeking is complete
            else:
                time.sleep(3)
            
            # Try reading multiple times for video files to get a valid frame
            max_attempts = 10 if camera_source == 'video_file' else 1
            ret, frame = False, None
            
            for attempt in range(max_attempts):
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Check if frame is not blank/black
                    if not self._is_blank_frame(frame):
                        self.logger.info(f"Successfully captured valid frame from camera {camera.id} on attempt {attempt + 1}")
                        break
                    else:
                        self.logger.debug(f"Attempt {attempt + 1}: Got blank frame, trying next")
                        if camera_source == 'video_file' and attempt < max_attempts - 1:
                            # Skip a few frames and try again
                            for _ in range(5):
                                cap.read()
                else:
                    self.logger.debug(f"Attempt {attempt + 1}: Failed to read frame")
                    break
            cap.release()
            
            if ret and frame is not None:
                # Resize frame to reasonable size for thumbnail
                height, width = frame.shape[:2]
                if width > 800:
                    new_width = 800
                    scale_ratio = new_width / width
                    new_height = int(height * scale_ratio)
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    # Convert to base64
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    self.logger.info(f"Successfully captured frame from camera {camera.id}")
                    return img_base64
                else:
                    self.logger.error(f"Failed to encode frame from camera {camera.id}")
                    return None
            else:
                self.logger.warning(f"Failed to read frame from camera {camera.id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error capturing frame from camera {camera.id}: {str(e)}")
            return None


# Create a global instance for use in views
rtsp_manager = RTSPConnectionManager()

# Backward compatibility functions
def generate_frames(camera_id, frame_rate=15, target_width=1280):
    """Backward compatible function that uses the class-based manager"""
    return rtsp_manager.generate_frames(camera_id, frame_rate, target_width)

def test_camera_connection(camera_id):
    """Test camera connection using the optimized manager"""
    camera = rtsp_manager.get_camera_by_id(camera_id)
    if not camera:
        return False
    return rtsp_manager.test_camera_connection(camera)

def get_all_cameras_status():
    """Get all cameras status using the optimized manager"""
    return rtsp_manager.get_all_cameras_status()

def capture_single_frame(camera_id):
    """Capture a single frame from camera for thumbnail generation"""
    try:
        camera = Camera.objects.get(id=camera_id)
        print(f"camera : {camera.id} - {camera.name}")
        return rtsp_manager.capture_single_frame(camera)
    except Camera.DoesNotExist:
        logger.error(f"Camera with ID {camera_id} not found")
        return None
    except Exception as e:
        logger.error(f"Error getting camera {camera_id}: {str(e)}")
        return None