import logging
from django.conf import settings
from pathlib import Path
from . models import SystemConfig

logger = logging.getLogger(__name__)

try:
    from ruamel.yaml import YAML
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    YAML_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import ruamel.yaml: {e}. Install with: pip install ruamel.yaml")
    YAML_AVAILABLE = False
    yaml = None

yaml_path = Path(settings.MEDIAMTX_ROOT) / 'mediamtx.yml'


def check_mediamtx_source(camera_name: str) -> bool:
    if not YAML_AVAILABLE:
        logger.error("Cannot check MediaMTX source: ruamel.yaml is not available")
        return False
    
    try:
        #logger.info(f"Checking MediaMTX source for camera '{camera_name}'")
        
        if not yaml_path.exists():
            logger.error(f"MediaMTX YAML file does not exist: {yaml_path}")
            return False
            
        with open(yaml_path, 'r') as file:
            data = yaml.load(file)
            
        if data is None:
            logger.error("Failed to load YAML data - file may be corrupted")
            return False
            
        if 'paths' not in data:
            data['paths'] = {}
            logger.info("Created 'paths' section in MediaMTX config") 
        
        if camera_name in data['paths']:
            #logger.info(f"Camera '{camera_name}' exists in MediaMTX config")
            return True
        else:
            #logger.info(f"Camera '{camera_name}' does not exist in MediaMTX config")
            return False
        
    except FileNotFoundError:
        logger.error(f"MediaMTX YAML file not found: {yaml_path}")
        return False
    except PermissionError:
        logger.error(f"Permission denied writing to MediaMTX YAML file: {yaml_path}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error adding MediaMTX source for {camera_name}: {type(e).__name__}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def add_mediamtx_source(camera_name: str, rtsp_url: str) -> bool:
    if not YAML_AVAILABLE:
        logger.error("Cannot add MediaMTX source: ruamel.yaml is not available")
        return False
    
    try:
        #logger.info(f"Adding MediaMTX source for camera '{camera_name}' with URL '{rtsp_url}'")
        
        if not yaml_path.exists():
            logger.error(f"MediaMTX YAML file does not exist: {yaml_path}")
            return False
            
        with open(yaml_path, 'r') as file:
            data = yaml.load(file)
            
        if data is None:
            logger.error("Failed to load YAML data - file may be corrupted")
            return False
            
        if 'paths' not in data:
            data['paths'] = {}
            logger.info("Created 'paths' section in MediaMTX config")
        
        if camera_name in data['paths']:
            logger.warning(f"Camera '{camera_name}' already exists in MediaMTX config")
            return False
        
        if rtsp_url.startswith("rtsp://"):

            data['paths'][camera_name] = {
                "source": rtsp_url,
                "rtspTransport": "automatic",
                "sourceOnDemand": "yes"
            }
            
            with yaml_path.open("w") as file:
                yaml.dump(data, file)
        else:
            pass
            
        #logger.info(f"Successfully added MediaMTX source for {camera_name} at {rtsp_url}")
        return True
        
    except FileNotFoundError:
        logger.error(f"MediaMTX YAML file not found: {yaml_path}")
        return False
    except PermissionError:
        logger.error(f"Permission denied writing to MediaMTX YAML file: {yaml_path}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error adding MediaMTX source for {camera_name}: {type(e).__name__}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def update_mediamtx_source(camera_name: str, rtsp_url: str, status: str) -> bool:
    
    if not YAML_AVAILABLE:
        logger.error("Cannot update MediaMTX source: ruamel.yaml is not available")
        return False

   
    try:
        #logger.info(f"Updating MediaMTX source for camera '{camera_name}' with URL '{rtsp_url}'")
        if rtsp_url.startswith("rtsp://"):
            if not yaml_path.exists():
                logger.error(f"MediaMTX YAML file does not exist: {yaml_path}")
                return False
                
            with open(yaml_path, 'r') as file:
                data = yaml.load(file)
                
            if data is None:
                logger.error("Failed to load YAML data - file may be corrupted")
                return False
            
            if 'paths' not in data or camera_name not in data['paths']:
                logger.error(f"Camera '{camera_name}' not found in MediaMTX config")
                return False
            
            data['paths'][camera_name]['source'] = rtsp_url
            logger.info("status is: " + str(status))

            if status == "True" or status == True:
                data['paths'][camera_name]['sourceOnDemand'] = 'no'
            elif status == "False" or status == False:
                data['paths'][camera_name]['sourceOnDemand'] = 'yes'

            logger.info(f"Setting sourceOnDemand for {camera_name} to {status}")
            
            with yaml_path.open("w") as file:
                yaml.dump(data, file)
                
            #logger.info(f"Successfully updated MediaMTX source for {camera_name} to {rtsp_url}")
            return True
        
    except FileNotFoundError:
        logger.error(f"MediaMTX YAML file not found: {yaml_path}")
        return False
    except PermissionError:
        logger.error(f"Permission denied writing to MediaMTX YAML file: {yaml_path}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error updating MediaMTX source for {camera_name}: {type(e).__name__}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def remove_mediamtx_source(camera_name: str) -> bool:
    if not YAML_AVAILABLE:
        logger.error("Cannot remove MediaMTX source: ruamel.yaml is not available")
        return False
    
    try:
        #logger.info(f"Removing MediaMTX source for camera '{camera_name}'")
        
        if not yaml_path.exists():
            logger.error(f"MediaMTX YAML file does not exist: {yaml_path}")
            return False
            
        with open(yaml_path, 'r') as file:
            data = yaml.load(file)
            
        if data is None:
            logger.error("Failed to load YAML data - file may be corrupted")
            return False
        
        if 'paths' not in data or camera_name not in data['paths']:
            logger.error(f"Camera '{camera_name}' not found in MediaMTX config")
            return False
        
        del data['paths'][camera_name]
        
        with yaml_path.open("w") as file:
            yaml.dump(data, file)
            
        #logger.info(f"Successfully removed MediaMTX source for {camera_name}")
        return True
        
    except FileNotFoundError:
        logger.error(f"MediaMTX YAML file not found: {yaml_path}")
        return False
    except PermissionError:
        logger.error(f"Permission denied writing to MediaMTX YAML file: {yaml_path}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error removing MediaMTX source for {camera_name}: {type(e).__name__}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def replace_mediamtx_source(camera_original: str, camera_name: str, rtsp_url: str) -> bool:
    if not YAML_AVAILABLE:
        logger.error("Cannot replace MediaMTX source: ruamel.yaml is not available")
        return False
    
    try:
        #logger.info(f"Replacing MediaMTX source '{camera_original}' with '{camera_name}' using URL '{rtsp_url}'")
        
        if not yaml_path.exists():
            logger.error(f"MediaMTX YAML file does not exist: {yaml_path}")
            return False
            
        with open(yaml_path, 'r') as file:
            data = yaml.load(file)
            
        if data is None:
            logger.error("Failed to load YAML data - file may be corrupted")
            return False
        
        if 'paths' not in data or camera_original not in data['paths']:
            logger.error(f"Original camera '{camera_original}' not found in MediaMTX config")
            return False
        
        # Check if new camera name already exists (and it's not the same as original)
        if camera_name != camera_original and camera_name in data['paths']:
            logger.error(f"New camera name '{camera_name}' already exists in MediaMTX config")
            return False
        
        # Replace the camera
        data['paths'][camera_name] = data['paths'].pop(camera_original)
        data['paths'][camera_name]['source'] = rtsp_url
        
        with yaml_path.open("w") as file:
            yaml.dump(data, file)
        
        #logger.info(f"Successfully replaced MediaMTX source {camera_original} with {camera_name} at {rtsp_url}")
        return True
        
    except FileNotFoundError:
        logger.error(f"MediaMTX YAML file not found: {yaml_path}")
        return False
    except PermissionError:
        logger.error(f"Permission denied writing to MediaMTX YAML file: {yaml_path}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error replacing MediaMTX source {camera_original} -> {camera_name}: {type(e).__name__}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def check_hls_stream_availability(camera_name: str, hls_host: str , hls_port: int = 8888) -> dict:
    """
    Check if HLS stream is available for a camera
    Returns dict with availability status and stream info
    """
    import requests
    from urllib.parse import quote
    
    # URL encode camera name to handle special characters
    encoded_camera_name = quote(camera_name)
    mediamtx = SystemConfig.objects.get(key='mediamtx_IP').value
    hls_url = f"{mediamtx}:{hls_port}/{encoded_camera_name}/index.m3u8"
    
    try:
        #logger.info(f"Checking HLS stream availability for camera '{camera_name}' at {hls_url}")
        
        # Make a HEAD request to check if the stream exists without downloading content
        response = requests.head(hls_url, timeout=5)
        
        if response.status_code == 200:
            logger.info(f"HLS stream available for camera '{camera_name}'")
            return {
                'available': True,
                'url': hls_url,
                'status_code': response.status_code,
                'content_type': response.headers.get('Content-Type', 'unknown'),
                'error': None
            }
        else:
            logger.warning(f"HLS stream not available for camera '{camera_name}' - Status: {response.status_code}")
            return {
                'available': False,
                'url': hls_url,
                'status_code': response.status_code,
                'content_type': None,
                'error': f"HTTP {response.status_code}"
            }
            
    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to MediaMTX HLS server at {hls_host}:{hls_port}")
        return {
            'available': False,
            'url': hls_url,
            'status_code': None,
            'content_type': None,
            'error': 'Connection refused - MediaMTX server may not be running'
        }
    except requests.exceptions.Timeout:
        logger.error(f"Timeout checking HLS stream for camera '{camera_name}'")
        return {
            'available': False,
            'url': hls_url,
            'status_code': None,
            'content_type': None,
            'error': 'Request timeout'
        }
    except Exception as e:
        logger.error(f"Unexpected error checking HLS stream for camera '{camera_name}': {type(e).__name__}: {e}")
        return {
            'available': False,
            'url': hls_url,
            'status_code': None,
            'content_type': None,
            'error': str(e)
        }