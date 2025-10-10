import asyncio
import cv2
import time
import threading
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from django.core.cache import cache
from .models import Camera
from typing import Optional, Dict, Any
import weakref

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures"""
    
    def __init__(self, failure_threshold=5, recovery_timeout=60, expected_exception=Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
            return result
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
            
            raise e

class AsyncRTSPManager:
    """Non-blocking RTSP manager with timeout handling and circuit breaker"""
    
    def __init__(self, max_workers=10, connection_timeout=10, read_timeout=5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.connection_timeout = connection_timeout
        self.read_timeout = read_timeout
        self.circuit_breakers = {}
        self.active_connections = weakref.WeakValueDictionary()
        self.logger = logging.getLogger(__name__)
        
    def get_circuit_breaker(self, camera_id):
        """Get or create circuit breaker for camera"""
        if camera_id not in self.circuit_breakers:
            self.circuit_breakers[camera_id] = CircuitBreaker()
        return self.circuit_breakers[camera_id]
    
    async def test_camera_connection_async(self, camera_id: int) -> Dict[str, Any]:
        """Async camera connection test with timeout"""
        try:
            camera = await self._get_camera_async(camera_id)
            if not camera:
                return {"success": False, "message": "Camera not found"}
            
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.executor, 
                    self._test_connection_sync, 
                    camera
                ),
                timeout=self.connection_timeout
            )
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Connection test timeout for camera {camera_id}")
            return {
                "success": False, 
                "message": f"Connection test timed out after {self.connection_timeout}s"
            }
        except Exception as e:
            self.logger.error(f"Connection test error for camera {camera_id}: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def _test_connection_sync(self, camera) -> Dict[str, Any]:
        """Synchronous connection test with circuit breaker"""
        circuit_breaker = self.get_circuit_breaker(camera.id)
        
        try:
            return circuit_breaker.call(self._do_connection_test, camera)
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def _do_connection_test(self, camera) -> Dict[str, Any]:
        """Actual connection test implementation"""
        rtsp_url = self._get_camera_url(camera)
        if not rtsp_url:
            raise Exception("No valid RTSP URL")
        
        self.logger.info(f"Testing connection to camera {camera.id} with URL: {rtsp_url}")
        
        # Create VideoCapture with FFMPEG backend and timeout properties
        cap = cv2.VideoCapture()
        
        # Set timeout properties before opening
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.connection_timeout * 1000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout * 1000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        try:
            # Try to open with FFMPEG backend first
            self.logger.debug(f"Attempting FFMPEG backend for camera {camera.id}")
            success = cap.open(rtsp_url, cv2.CAP_FFMPEG)
            if not success:
                # Fallback to default backend
                self.logger.debug(f"FFMPEG failed, trying default backend for camera {camera.id}")
                cap.release()
                cap = cv2.VideoCapture()
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.connection_timeout * 1000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout * 1000)
                success = cap.open(rtsp_url)
                
            if not success:
                self.logger.error(f"Failed to open stream for camera {camera.id}")
                raise Exception("Failed to open RTSP stream with any backend")
            
            self.logger.debug(f"Stream opened successfully for camera {camera.id}, testing frame read")
            
            # Quick frame read test
            ret, frame = cap.read()
            if not ret or frame is None:
                self.logger.error(f"Failed to read frame from camera {camera.id}")
                raise Exception("Failed to read frame from stream")
            
            # Get stream properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Connection test successful for camera {camera.id}: {width}x{height} @ {fps}fps")
            
            return {
                "success": True,
                "message": "Connection successful",
                "resolution": f"{width}x{height}",
                "fps": fps if fps > 0 else 25,
                "camera_id": camera.id
            }
            
        except Exception as e:
            self.logger.error(f"Connection test failed for camera {camera.id}: {str(e)}")
            raise e
        finally:
            try:
                cap.release()
            except:
                pass
    
    async def get_camera_status_async(self, camera_id: int) -> Dict[str, Any]:
        """Get camera status asynchronously"""
        cache_key = f"camera_status_{camera_id}"
        cached_status = cache.get(cache_key)
        
        if cached_status:
            return cached_status
        
        try:
            status = await self.test_camera_connection_async(camera_id)
            
            # Cache status for 30 seconds
            cache.set(cache_key, status, 30)
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting camera status {camera_id}: {str(e)}")
            return {"success": False, "message": "Status check failed"}
    
    async def _get_camera_async(self, camera_id: int) -> Optional[Camera]:
        """Get camera object asynchronously"""
        loop = asyncio.get_event_loop()
        try:
            camera = await loop.run_in_executor(
                self.executor,
                lambda: Camera.objects.get(id=camera_id, is_active=True)
            )
            return camera
        except Camera.DoesNotExist:
            return None
    
    def _get_camera_url(self, camera) -> Optional[str]:
        """Get RTSP URL for camera"""
        rtsp_url = camera.rtsp_url
        if not rtsp_url:
            return None
        
        if camera.username and camera.password:
            clean_url = rtsp_url.replace('rtsp://', '')
            if '@' in clean_url:
                clean_url = clean_url.split('@', 1)[1]
            rtsp_url = f"rtsp://{camera.username}:{camera.password}@{clean_url}"
        
        return rtsp_url
    
    async def generate_frames_async(self, camera_id: int, frame_rate: int = 15):
        """Async frame generator with timeout handling"""
        camera = await self._get_camera_async(camera_id)
        if not camera:
            async for frame in self._generate_fallback_frames_async(frame_rate, "Camera not found"):
                yield frame
            return
        
        rtsp_url = self._get_camera_url(camera)
        if not rtsp_url:
            async for frame in self._generate_fallback_frames_async(frame_rate, "No RTSP URL"):
                yield frame
            return
        
        circuit_breaker = self.get_circuit_breaker(camera_id)
        
        try:
            # Check circuit breaker state
            if circuit_breaker.state == 'OPEN':
                async for frame in self._generate_fallback_frames_async(frame_rate, "Circuit breaker OPEN"):
                    yield frame
                return
            
            loop = asyncio.get_event_loop()
            frame_generator = await loop.run_in_executor(
                self.executor,
                self._create_frame_generator,
                camera_id, rtsp_url, frame_rate
            )
            
            async for frame in frame_generator:
                yield frame
                
        except Exception as e:
            self.logger.error(f"Frame generation error for camera {camera_id}: {str(e)}")
            async for frame in self._generate_fallback_frames_async(frame_rate, str(e)):
                yield frame
    
    def _create_frame_generator(self, camera_id: int, rtsp_url: str, frame_rate: int):
        """Create synchronous frame generator with timeout"""
        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.connection_timeout * 1000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout * 1000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        try:
            if not cap.open(rtsp_url, cv2.CAP_FFMPEG):
                raise Exception("Failed to open stream")
            
            consecutive_failures = 0
            max_failures = 5
            
            while consecutive_failures < max_failures:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0
                
                # Encode frame
                ret_encode, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret_encode:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                time.sleep(1 / frame_rate)
                
        finally:
            cap.release()
    
    async def _generate_fallback_frames_async(self, frame_rate: int, error_message: str = "No Video"):
        """Generate fallback frames asynchronously"""
        no_video_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        no_video_frame[:] = (45, 45, 45)
        
        cv2.rectangle(no_video_frame, (5, 5), (315, 235), (80, 80, 80), 2)
        cv2.putText(no_video_frame, 'NO VIDEO', (85, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(no_video_frame, 'SIGNAL', (105, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.circle(no_video_frame, (160, 180), 8, (0, 0, 255), -1)
        cv2.putText(no_video_frame, error_message[:20], (60, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        while True:
            ret, buffer = cv2.imencode('.jpg', no_video_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            await asyncio.sleep(1 / frame_rate)
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=False)

# Global async manager instance
async_rtsp_manager = AsyncRTSPManager()