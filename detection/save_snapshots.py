import os
import cv2
import logging

from torch.utils.data import DataChunk
from django.utils import timezone
from django.conf import settings
from core.models import Camera, Detection, Alert,SystemConfig, PPEChecks
import time
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

import requests


MEDIA_ROOT = settings.MEDIA_ROOT
BASE_DIR = settings.BASE_DIR
# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_frameid(camera_id, current_timestamp=None):
    """
    Generate a hierarchical timestamp-based frameid
    Format: {camera_id}_{YYYYMMDD}_{HHMMSS}_{microseconds}
    Example: CAM001_20250917_143052_123456
    """
    if current_timestamp is None:
        current_timestamp = timezone.now()
    
    # Format camera ID to ensure consistent length
    camera_str = f"CAM{str(camera_id).zfill(3)}"
    
    # Extract date, time, and microseconds
    date_str = current_timestamp.strftime('%Y%m%d')
    time_str = current_timestamp.strftime('%H%M%S')
    microsec_str = str(current_timestamp.microsecond).zfill(6)
    
    frameid = f"{camera_str}_{date_str}_{time_str}_{microsec_str}"
    
    return frameid

# Function to save a snapshot and record in the database asynchronously
def save_detection_ppecheck(frame, cameraid, detection_id, ppe_data):
    try:
        camera = Camera.objects.get(id=cameraid)
        cameraname = camera.name
            
        current_timestamp = timezone.now()
        timestamp = current_timestamp.strftime('%d-%m-%y-%H-%M-%S')
        
        # Generate hierarchical frameid for PPE check
        frameid = generate_frameid(cameraid, current_timestamp)

        snapshot_filename = f"{cameraname} - PPECheck - {timestamp}.jpg"
        snapshot_path = os.path.join(settings.MEDIA_ROOT, 'Snapshot/', snapshot_filename)

        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
        
        if not cv2.imwrite(snapshot_path, frame):
            raise ValueError(f"Failed to save snapshot to {snapshot_path}")
        
        try:
            # Save to PPEChecks table with correct field mapping
            ppe_check = PPEChecks(
                camera=camera,  # Use camera object, not cameraid
                Detectionid_id=detection_id,  # ForeignKey field
                trackid=ppe_data.get('trackid'),
                has_helmet=ppe_data.get('has_helmet', False),
                has_vest=ppe_data.get('has_vest', False),
                has_goggles=ppe_data.get('has_goggles', False),
                has_gloves=ppe_data.get('has_gloves', False),
                has_shoes=ppe_data.get('has_shoes', False),
                details=ppe_data.get('details', {})
            )
            ppe_check.save()
            
            logger.info(f"PPE Check recorded for camera: {cameraname} with snapshot: {snapshot_filename} at: {current_timestamp}")
        except Exception as e:
            logger.error(f"Database save failed: {e}")
            raise
    except Exception as e:
        logger.error(f"Error in save_detection_ppecheck: {e}")

# Function to save a snapshot and record in the database asynchronously
def save_detection_intrusioncheck(frame, cameraid, detection_id, intrusion_data):
    try:
        camera = Camera.objects.get(id=cameraid)
        cameraname = camera.name
            
        current_timestamp = timezone.now()
        timestamp = current_timestamp.strftime('%d-%m-%y-%H-%M-%S')
        
        # Generate hierarchical frameid for PPE check
        frameid = generate_frameid(cameraid, current_timestamp)

        snapshot_filename = f"{cameraname} - PPECheck - {timestamp}.jpg"
        snapshot_path = os.path.join(settings.MEDIA_ROOT, 'Snapshot/', snapshot_filename)

        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
        
        if not cv2.imwrite(snapshot_path, frame):
            raise ValueError(f"Failed to save snapshot to {snapshot_path}")
        
        try:
            # Save to PPEChecks table with correct field mapping
            ppe_check = PPEChecks(
                camera=camera,  # Use camera object, not cameraid
                Detectionid_id=detection_id,  # ForeignKey field
                trackid=ppe_data.get('trackid'),
                has_helmet=ppe_data.get('has_helmet', False),
                has_vest=ppe_data.get('has_vest', False),
                has_goggles=ppe_data.get('has_goggles', False),
                has_gloves=ppe_data.get('has_gloves', False),
                has_shoes=ppe_data.get('has_shoes', False),
                details=ppe_data.get('details', {})
            )
            ppe_check.save()
            
            logger.info(f"PPE Check recorded for camera: {cameraname} with snapshot: {snapshot_filename} at: {current_timestamp}")
        except Exception as e:
            logger.error(f"Database save failed: {e}")
            raise
    except Exception as e:
        logger.error(f"Error in save_detection_ppecheck: {e}")

def save_snapshot_and_record(frame, cameraid, message_body, detection_type, object_count):

    try:
        camera = Camera.objects.get(id=cameraid)
        cameraname = camera.name

        message = message_body
        #message = message.replace("<count>", str(unique_count))
        #message = message.replace("<cameraname", cameraname)
        #message = message.replace("<cameralocation", camera_location)
        #message = message.replace("<time>", str(current_time_pass))
            
        current_timestamp = timezone.now()
        timestamp = current_timestamp.strftime('%d-%m-%y-%H-%M-%S')
        
        # Generate hierarchical frameid
        frameid = generate_frameid(cameraid, current_timestamp)

        snapshot_filename = f"{cameraname} - {timestamp}.jpg"
        snapshot_path = os.path.join(settings.MEDIA_ROOT, 'Snapshot/', snapshot_filename)

        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
        
        if not cv2.imwrite(snapshot_path, frame):
            raise ValueError(f"Failed to save snapshot to {snapshot_path}")
        
        try:
            # Save to Detection table with correct field mapping
            detection = Detection(
                camera=camera,  # Use camera object, not cameraid
                frameid=frameid,  # Use hierarchical timestamp frameid
                description=message,  # Use description field, not message
                detection_type=detection_type,
                object_count=object_count,
                annotated_snapshot_path=f'Snapshot/{snapshot_filename}'  # Use annotated_snapshot_path field
            )
            detection.save()
            
            # Create Dummy ticket to saras 
            data = {
                        "account_code": "8888",
                        "event": "1120",
                        "partition": "01",
                        "zone": "001",
                        "extra_message": message,
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "is_heartbeat": False,
                    }
                    
          
            response_data = send_alert_to_saras(cameraid, data)

            tiket_saras = None
            event_saras = None
            if isinstance(response_data, dict) and 'message' in response_data:
                tiket_saras = response_data['message'].get('TicketNo', '')
                event_saras = response_data['message'].get('EventName', '')
           
            alert = Alert(
                camera=camera,
                detection=detection,
                severity='medium',
                message=f"TicketNo:{tiket_saras}, EventName:{event_saras}: message:{message}" if tiket_saras else message,
                is_acknowledged=False
            )
            alert.save()
            
            logger.info(f"New incident on camera: {cameraname} with snapshot: {snapshot_filename} and alert created at: {current_timestamp}")
            
            # Return the detection ID for use in related records
            return detection.pk
            
        except Exception as e:
            logger.error(f"Database save failed: {e}")
            raise
    except Exception as e:
        logger.error(f"Error in save snapshot_and_record: {e}")
        return None

"""
# Function to save a snapshot and record in the database asynchronously
def save_record(unique_id, unique_count, frame, cameraid, cameraname,camera_location,current_time_pass, message_body):

    try:    
        message = message_body
        message = message.replace("<count>", str(unique_count))
        message = message.replace("<cameraname", cameraname)
        message = message.replace("<cameralocation", camera_location)
        message = message.replace("<time>", str(current_time_pass))
            
        current_timestamp = timezone.now()
        timestamp = current_timestamp.strftime('%d-%m-%y-%H-%M-%S')
        snapshot_filename = f"{cameraname} - {timestamp}.jpg"
        snapshot_path = os.path.join(settings.MEDIA_ROOT, 'Snapshot/objdetection/', snapshot_filename)

        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
        
        if not cv2.imwrite(snapshot_path, frame):
            raise ValueError(f"Failed to save snapshot to {snapshot_path}")
        
        try:
            obj_detection = ObjDetection(
                count=unique_count,
                image=snapshot_filename,
                cameraid_id=cameraid,
                unique_id=unique_id,
                message=message
            )
            obj_detection.save()
            logger.info(f"Recorded {unique_count} person(s) with snapshot: {snapshot_filename}")
        except Exception as e:
            logger.error(f"database save failed: {e}")
            raise
    except Exception as e:
        logger.error(f"Error in save snapshot_and_record: {e}")

# Function to save a snapshot and record in the database asynchronously
def save_video_record(unique_id, unique_count, frame, cameraid, cameraname,camera_location,current_time_pass, message_body,video_duration=10, fps=15):

    try:    
        message = message_body
        message = message.replace("<count>", str(unique_count))
        message = message.replace("<cameraname", cameraname)
        message = message.replace("<cameralocation", camera_location)
        message = message.replace("<time>", str(current_time_pass))
            
        current_timestamp = timezone.now()
        timestamp = current_timestamp.strftime('%d-%m-%y-%H-%M-%S')
        video_filename = f"{cameraname.replace(' ', '_')}_{timestamp}.mp4"
        snapshot_path = os.path.join(settings.MEDIA_ROOT, 'Snapshot/objdetection/', video_filename)

        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)

        #define video codec and initialize video writer
        height, width, _ =frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(snapshot_path, fourcc, fps, (width, height))

        # Start recording for the specified duration
        start_time = time.time()
        while True:
            ret, frame = camera.read()
            if not ret:
                logger.error("Failed to read frame during recording.")
                break

            # Write the frame to the video
            out.write(frame)

            # Check if recording time is up
            elapsed_time = time.time() - start_time
            if elapsed_time >= video_duration:
                break

        # Release video writer
        out.release()
        
        
        try:
            obj_detection = ObjDetection(
                count=unique_count,
                image=video_filename,
                cameraid_id=cameraid,
                unique_id=unique_id,
                message=message
            )
            obj_detection.save()
            logger.info(f"Recorded {unique_count} person(s) with snapshot: {video_filename}")
        except Exception as e:
            logger.error(f"database save failed: {e}")
            raise
    except Exception as e:
        logger.error(f"Error in save snapshot_and_record: {e}")
"""

def send_alert_to_saras(cameraid, data: Dict[str, Any]) -> bool:
    """
    Send alert to SARAS API with proper error handling and logging.
    
    Args:
        frame: Camera frame data
        cameraid: Camera identifier  
        data: Alert data dictionary containing required fields
        
    Returns:
        bool: True if successful, False otherwise
    """
    response = []

    # Get configuration values
    url = SystemConfig.get_value('saras_api_url')
    token = SystemConfig.get_value('saras_api_token')
    
    if not url or not token:
        logging.error("Missing SARAS API configuration (URL or token)")
        return False
    
    # Validate required fields
    required_fields = ['account_code', 'partition', 'zone', 'extra_message', 'timestamp', 'is_heartbeat']
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        logging.error(f"Missing required fields in data: {missing_fields}")
        return False
    
    # Build payload - no need for intermediate variables
    payload = {
        "account_code": data['account_code'],
        "event": data.get('event', "1121"),  # Allow override, default to 1121
        "partition": data['partition'],
        "zone": data['zone'],
        "extra_message": data['extra_message'],
        "timestamp": data['timestamp'],
        "is_heartbeat": data['is_heartbeat']
    }
    
    # Headers matching the curl example exactly
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"token {token}",  # Fixed: use 'token' not 'Bearer'
    }
    
    try:
        # Add timeout to prevent hanging
        response = requests.post(
            url, 
            json=payload, 
            headers=headers,
            timeout=30  # 30 second timeout
        )

        print(f"Sending alert to SARAS: {payload}")
        print(f"Sending Header to SARAS: {headers}")

        # Check for HTTP errors
        response.raise_for_status()
        
        # Log successful response
        response_data = response.json() if response.content else {}
       #logging.info(f"Alert sent to SARAS successfully. Response: {response_data}")
        #print(f"Alert sent to SARAS successfully: {response_data}")
        
        return response_data if response_data else True  # Return response data if available, otherwise True
        
    except requests.exceptions.Timeout:
        error_msg = f"SARAS API request timed out for camera {cameraid}"
        logging.error(error_msg)
        print(f"Error: {error_msg}")
        return False
        
    except requests.exceptions.ConnectionError:
        error_msg = f"Failed to connect to SARAS API: {url}"
        logging.error(error_msg)
        print(f"Error: {error_msg}")
        return False
        
    except requests.exceptions.HTTPError as e:
        error_msg = f"SARAS API returned HTTP error {e.response.status_code}: {e.response.text}"
        logging.error(error_msg)
        print(f"Error: {error_msg}")
        return False
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Error sending alert to SARAS: {str(e)}"
        logging.error(error_msg)
        print(f"Error: {error_msg}")
        return False
        
    except Exception as e:
        error_msg = f"Unexpected error sending alert to SARAS: {str(e)}"
        logging.error(error_msg)
        print(f"Error: {error_msg}")
        return False
