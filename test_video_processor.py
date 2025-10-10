#!/usr/bin/env python3
"""
Test script for SOCA Edge Video Processor
Tests the integration with Django models
"""

import os
import sys
import django
from pathlib import Path

# Setup Django
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'socaedge.settings')
django.setup()

from core.models import Camera, Detection, Alert, SystemConfig
from django.utils import timezone
from datetime import timedelta

def test_django_integration():
    """Test Django model integration"""
    print("🧪 Testing SOCA Edge Video Processor Django Integration")
    print("=" * 60)
    
    # Test 1: Check database connectivity
    print("\n📊 Test 1: Database Connectivity")
    try:
        camera_count = Camera.objects.count()
        detection_count = Detection.objects.count()
        alert_count = Alert.objects.count()
        config_count = SystemConfig.objects.count()
        
        print(f"✅ Database accessible")
        print(f"   • Cameras: {camera_count}")
        print(f"   • Detections: {detection_count}")
        print(f"   • Alerts: {alert_count}")
        print(f"   • Configs: {config_count}")
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    
    # Test 2: Load active cameras
    print("\n📹 Test 2: Load Active Cameras")
    try:
        active_cameras = Camera.objects.filter(is_active=True)
        print(f"✅ Found {active_cameras.count()} active cameras:")
        
        for camera in active_cameras:
            print(f"   • {camera.name} - {camera.rtsp_url}")
            roi = camera.get_roi_coordinates()
            print(f"     ROI: {len(roi) if roi else 0} coordinates")
            
    except Exception as e:
        print(f"❌ Camera loading error: {e}")
        return False
    
    # Test 3: System configuration
    print("\n⚙️  Test 3: System Configuration")
    try:
        sensitivity = SystemConfig.get_value('detection_sensitivity', '0.7')
        recording = SystemConfig.get_value('recording_enabled', 'true')
        email = SystemConfig.get_value('alert_email', 'admin@example.com')
        
        print(f"✅ Configuration loaded:")
        print(f"   • Detection sensitivity: {sensitivity}")
        print(f"   • Recording enabled: {recording}")
        print(f"   • Alert email: {email}")
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    # Test 4: Create test detection
    print("\n🔍 Test 4: Create Test Detection")
    try:
        if active_cameras.exists():
            test_camera = active_cameras.first()
            
            # Create test detection
            test_detection = Detection.objects.create(
                camera=test_camera,
                description="Test detection from video processor",
                annotated_snapshot_path="test/test_snapshot.jpg",
                timestamp=timezone.now()
            )
            
            print(f"✅ Test detection created:")
            print(f"   • ID: {test_detection.id}")
            print(f"   • Camera: {test_detection.camera.name}")
            print(f"   • Description: {test_detection.description}")
            
            # Create test alert
            test_alert = Alert.objects.create(
                camera=test_camera,
                detection=test_detection,
                severity='medium',
                message="Test alert from video processor integration",
                is_acknowledged=False,
                created_at=timezone.now()
            )
            
            print(f"✅ Test alert created:")
            print(f"   • ID: {test_alert.id}")
            print(f"   • Severity: {test_alert.severity}")
            print(f"   • Message: {test_alert.message}")
            
            # Clean up test records
            test_alert.delete()
            test_detection.delete()
            print("✅ Test records cleaned up")
            
        else:
            print("⚠️  No active cameras found for testing")
            
    except Exception as e:
        print(f"❌ Detection/Alert creation error: {e}")
        return False
    
    # Test 5: Video processor import
    print("\n🎥 Test 5: Video Processor Import")
    try:
        from video_processor.video_processor_standalone import VideoProcessor
        
        processor = VideoProcessor()
        print(f"✅ Video processor imported successfully")
        print(f"   • Detection sensitivity: {processor.detection_sensitivity}")
        print(f"   • Recording enabled: {processor.recording_enabled}")
        print(f"   • Max concurrent streams: {processor.max_concurrent_streams}")
        print(f"   • Snapshot directory: {processor.snapshot_dir}")
        
        # Test camera loading
        if processor.load_cameras():
            print(f"✅ Loaded {len(processor.cameras)} cameras into processor")
            for cam_id, cam_info in processor.cameras.items():
                print(f"   • Camera {cam_id}: {cam_info['name']}")
        else:
            print("⚠️  No cameras loaded into processor")
            
    except Exception as e:
        print(f"❌ Video processor import error: {e}")
        return False
    
    # Test 6: Recent data analysis
    print("\n📈 Test 6: Recent Data Analysis")
    try:
        # Recent detections (last 24 hours)
        recent_detections = Detection.objects.filter(
            timestamp__gte=timezone.now() - timedelta(hours=24)
        )
        
        # Unacknowledged alerts
        unack_alerts = Alert.objects.filter(is_acknowledged=False)
        
        print(f"✅ Data analysis:")
        print(f"   • Recent detections (24h): {recent_detections.count()}")
        print(f"   • Unacknowledged alerts: {unack_alerts.count()}")
        
        if recent_detections.exists():
            latest = recent_detections.latest('timestamp')
            print(f"   • Latest detection: {latest.description} ({latest.timestamp})")
        
        if unack_alerts.exists():
            critical_alerts = unack_alerts.filter(severity='critical').count()
            high_alerts = unack_alerts.filter(severity='high').count()
            print(f"   • Critical alerts: {critical_alerts}")
            print(f"   • High severity alerts: {high_alerts}")
            
    except Exception as e:
        print(f"❌ Data analysis error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Video processor is ready for Django integration.")
    print("\n🚀 To start the video processor:")
    print("   • Standalone: python video_processor/video_processor_standalone.py")
    print("   • Django command: python manage.py start_video_processor")
    
    return True

def show_system_status():
    """Show current system status"""
    print("\n📊 Current System Status:")
    print("-" * 30)
    
    try:
        # Camera status
        total_cameras = Camera.objects.count()
        active_cameras = Camera.objects.filter(is_active=True).count()
        print(f"📹 Cameras: {active_cameras}/{total_cameras} active")
        
        # Recent activity
        today_detections = Detection.objects.filter(
            timestamp__date=timezone.now().date()
        ).count()
        unack_alerts = Alert.objects.filter(is_acknowledged=False).count()
        
        print(f"🔍 Today's detections: {today_detections}")
        print(f"🚨 Unacknowledged alerts: {unack_alerts}")
        
        # System config
        email = SystemConfig.get_value('alert_email', 'Not configured')
        sensitivity = SystemConfig.get_value('detection_sensitivity', 'Default')
        print(f"📧 Alert email: {email}")
        print(f"🎯 Detection sensitivity: {sensitivity}")
        
    except Exception as e:
        print(f"❌ Error getting system status: {e}")

if __name__ == "__main__":
    # Run integration tests
    success = test_django_integration()
    
    if success:
        show_system_status()
    else:
        print("\n❌ Integration tests failed. Please check the errors above.")
        sys.exit(1)