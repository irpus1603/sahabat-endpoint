"""
Django management command to reset and clean sample data
Usage: python manage.py reset_sample_data
"""

from django.core.management.base import BaseCommand
from core.models import Camera, Detection, Alert

class Command(BaseCommand):
    help = 'Reset sample data for testing'

    def add_arguments(self, parser):
        parser.add_argument(
            '--activate-cameras',
            action='store_true',
            help='Activate first 4 cameras',
        )
        parser.add_argument(
            '--clear-old-data',
            action='store_true',
            help='Clear excessive detections and alerts',
        )

    def handle(self, *args, **options):
        if options['activate_cameras']:
            # Activate first 4 cameras
            cameras = Camera.objects.all()[:4]
            for camera in cameras:
                camera.is_active = True
                camera.save()
                self.stdout.write(f'✅ Activated: {camera.name}')
            
            # Deactivate the rest
            remaining = Camera.objects.all()[4:]
            for camera in remaining:
                camera.is_active = False
                camera.save()
                self.stdout.write(f'❌ Deactivated: {camera.name}')
        
        if options['clear_old_data']:
            # Keep only recent 20 detections
            recent_detections = Detection.objects.order_by('-timestamp')[:20]
            Detection.objects.exclude(id__in=[d.id for d in recent_detections]).delete()
            
            # Keep only recent 10 alerts
            recent_alerts = Alert.objects.order_by('-created_at')[:10]
            Alert.objects.exclude(id__in=[a.id for a in recent_alerts]).delete()
            
            self.stdout.write('🧹 Cleaned old data')
        
        # Show current status
        total_cameras = Camera.objects.count()
        active_cameras = Camera.objects.filter(is_active=True).count()
        total_detections = Detection.objects.count()
        total_alerts = Alert.objects.count()
        unack_alerts = Alert.objects.filter(is_acknowledged=False).count()
        
        self.stdout.write('\n📊 Current Status:')
        self.stdout.write(f'   Cameras: {active_cameras}/{total_cameras} active')
        self.stdout.write(f'   Detections: {total_detections}')
        self.stdout.write(f'   Alerts: {unack_alerts}/{total_alerts} unacknowledged')