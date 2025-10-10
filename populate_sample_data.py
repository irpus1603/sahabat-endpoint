#!/usr/bin/env python
"""
Sample data population script for SOCA Edge Surveillance System
Run this to populate the SQLite database with test data
"""

import os
import sys
import django
from datetime import datetime, timedelta
from django.utils import timezone

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'socaedge.settings')
django.setup()

from core.models import Camera, Detection, Alert, SystemConfig

def create_sample_cameras():
    """Create sample camera configurations"""
    cameras = [
        {
            'name': 'Main Entrance Camera',
            'rtsp_url': 'rtsp://192.168.1.100:554/stream1',
            'username': 'admin',
            'password': 'admin123',
            'is_active': True,
            'roi_coordinates': '[[100,100],[400,100],[400,300],[100,300]]'
        },
        {
            'name': 'Parking Lot Camera',
            'rtsp_url': 'rtsp://192.168.1.101:554/stream1',
            'username': 'admin',
            'password': 'admin123',
            'is_active': True,
            'roi_coordinates': '[[50,50],[600,50],[600,400],[50,400]]'
        },
        {
            'name': 'Side Gate Camera',
            'rtsp_url': 'rtsp://192.168.1.102:554/stream1',
            'username': 'admin',
            'password': 'admin123',
            'is_active': True,
            'roi_coordinates': '[[150,150],[500,150],[500,350],[150,350]]'
        },
        {
            'name': 'Back Yard Camera',
            'rtsp_url': 'rtsp://192.168.1.103:554/stream1',
            'username': 'admin',
            'password': 'admin123',
            'is_active': False,  # This one is offline
            'roi_coordinates': '[[80,80],[520,80],[520,320],[80,320]]'
        },
        {
            'name': 'Emergency Exit Camera',
            'rtsp_url': 'rtsp://192.168.1.104:554/stream1',
            'username': 'admin',
            'password': 'admin123',
            'is_active': True,
            'roi_coordinates': '[[200,200],[450,200],[450,350],[200,350]]'
        }
    ]
    
    created_cameras = []
    for camera_data in cameras:
        camera, created = Camera.objects.get_or_create(
            name=camera_data['name'],
            defaults=camera_data
        )
        if created:
            print(f"✓ Created camera: {camera.name}")
        else:
            print(f"- Camera already exists: {camera.name}")
        created_cameras.append(camera)
    
    return created_cameras

def create_sample_detections(cameras):
    """Create sample detection records"""
    detection_data = [
        {
            'description': 'Person detected at main entrance',
            'annotated_snapshot_path': '/media/snapshots/main_entrance_001.jpg',
            'camera': cameras[0],  # Main Entrance
            'hours_ago': 1
        },
        {
            'description': 'Vehicle movement in parking lot',
            'annotated_snapshot_path': '/media/snapshots/parking_lot_001.jpg',
            'camera': cameras[1],  # Parking Lot
            'hours_ago': 2
        },
        {
            'description': 'Motion detected at side gate',
            'annotated_snapshot_path': '/media/snapshots/side_gate_001.jpg',
            'camera': cameras[2],  # Side Gate
            'hours_ago': 3
        },
        {
            'description': 'Person loitering detected',
            'annotated_snapshot_path': '/media/snapshots/parking_lot_002.jpg',
            'camera': cameras[1],  # Parking Lot
            'hours_ago': 4
        },
        {
            'description': 'Emergency exit accessed',
            'annotated_snapshot_path': '/media/snapshots/emergency_exit_001.jpg',
            'camera': cameras[4],  # Emergency Exit
            'hours_ago': 6
        },
        {
            'description': 'Multiple persons detected',
            'annotated_snapshot_path': '/media/snapshots/main_entrance_002.jpg',
            'camera': cameras[0],  # Main Entrance
            'hours_ago': 8
        }
    ]
    
    created_detections = []
    for detection_info in detection_data:
        timestamp = timezone.now() - timedelta(hours=detection_info['hours_ago'])
        
        detection, created = Detection.objects.get_or_create(
            description=detection_info['description'],
            annotated_snapshot_path=detection_info['annotated_snapshot_path'],
            camera=detection_info['camera'],
            defaults={'timestamp': timestamp}
        )
        
        if created:
            print(f"✓ Created detection: {detection.description[:50]}...")
        else:
            print(f"- Detection already exists: {detection.description[:50]}...")
        created_detections.append(detection)
    
    return created_detections

def create_sample_alerts(cameras, detections):
    """Create sample alert records"""
    alert_data = [
        {
            'camera': cameras[1],  # Parking Lot
            'detection': detections[3],  # Person loitering
            'severity': 'critical',
            'message': 'Suspicious loitering detected in parking lot. Person has been in the area for over 10 minutes.',
            'is_acknowledged': False,
            'minutes_ago': 5
        },
        {
            'camera': cameras[0],  # Main Entrance
            'detection': detections[0],  # Person detected
            'severity': 'high',
            'message': 'Unauthorized access attempt at main entrance after business hours.',
            'is_acknowledged': False,
            'minutes_ago': 15
        },
        {
            'camera': cameras[3],  # Back Yard (offline)
            'detection': None,
            'severity': 'medium',
            'message': 'Camera offline: Back Yard Camera has been disconnected for 15 minutes.',
            'is_acknowledged': False,
            'minutes_ago': 30
        },
        {
            'camera': cameras[4],  # Emergency Exit
            'detection': detections[4],  # Emergency exit accessed
            'severity': 'high',
            'message': 'Emergency exit accessed outside of emergency drill schedule.',
            'is_acknowledged': True,
            'minutes_ago': 120
        },
        {
            'camera': cameras[2],  # Side Gate
            'detection': detections[2],  # Motion at side gate
            'severity': 'low',
            'message': 'Motion detected at side gate. Likely wildlife or wind movement.',
            'is_acknowledged': True,
            'minutes_ago': 180
        }
    ]
    
    created_alerts = []
    for alert_info in alert_data:
        timestamp = timezone.now() - timedelta(minutes=alert_info['minutes_ago'])
        
        alert, created = Alert.objects.get_or_create(
            camera=alert_info['camera'],
            message=alert_info['message'],
            defaults={
                'detection': alert_info['detection'],
                'severity': alert_info['severity'],
                'is_acknowledged': alert_info['is_acknowledged'],
                'created_at': timestamp
            }
        )
        
        if created:
            print(f"✓ Created alert: {alert.severity.upper()} - {alert.message[:50]}...")
        else:
            print(f"- Alert already exists: {alert.severity.upper()} - {alert.message[:50]}...")
        created_alerts.append(alert)
    
    return created_alerts

def create_system_config():
    """Create sample system configuration"""
    config_data = [
        {
            'key': 'detection_sensitivity',
            'value': '0.7',
            'category': 'detection'
        },
        {
            'key': 'recording_enabled',
            'value': 'true',
            'category': 'recording'
        },
        {
            'key': 'max_recording_days',
            'value': '30',
            'category': 'recording'
        },
        {
            'key': 'alert_email',
            'value': 'supriyadi1@ioh.co.id',
            'category': 'notifications'
        },
        {
            'key': 'alert_phone',
            'value': '0816769003',
            'category': 'notifications'
        },
        {
            'key': 'system_name',
            'value': 'SOCA Edge Surveillance',
            'category': 'general'
        },
        {
            'key': 'timezone',
            'value': 'Asia/Jakarta',
            'category': 'general'
        },
        {
            'key': 'auto_cleanup_enabled',
            'value': 'true',
            'category': 'maintenance'
        },
        {
            'key': 'backup_enabled',
            'value': 'true',
            'category': 'maintenance'
        },
        {
            'key': 'max_concurrent_streams',
            'value': '8',
            'category': 'performance'
        }
    ]
    
    created_configs = []
    for config_info in config_data:
        config, created = SystemConfig.objects.get_or_create(
            key=config_info['key'],
            defaults={
                'value': config_info['value'],
                'category': config_info['category']
            }
        )
        
        if created:
            print(f"✓ Created config: {config.key} = {config.value}")
        else:
            print(f"- Config already exists: {config.key} = {config.value}")
        created_configs.append(config)
    
    return created_configs

def main():
    """Main function to populate sample data"""
    print("🚀 Populating SOCA Edge Surveillance System with sample data...")
    print("=" * 60)
    
    # Check if data already exists
    if Camera.objects.exists():
        print("⚠️  Sample data already exists!")
        response = input("Do you want to continue and add more data? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    print("\n📹 Creating sample cameras...")
    cameras = create_sample_cameras()
    
    print(f"\n🔍 Creating sample detections...")
    detections = create_sample_detections(cameras)
    
    print(f"\n🚨 Creating sample alerts...")
    alerts = create_sample_alerts(cameras, detections)
    
    print(f"\n⚙️  Creating system configuration...")
    configs = create_system_config()
    
    print("\n" + "=" * 60)
    print("✅ Sample data population completed!")
    print(f"   • {len(cameras)} cameras created")
    print(f"   • {len(detections)} detections created")
    print(f"   • {len(alerts)} alerts created")
    print(f"   • {len(configs)} config entries created")
    print("\n🌐 You can now run: python manage.py runserver")
    print("   Then visit: http://127.0.0.1:8000/")

if __name__ == '__main__':
    main()