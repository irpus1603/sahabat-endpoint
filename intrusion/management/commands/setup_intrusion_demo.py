from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from core.models import Camera, Detection, SystemConfig, intrusionChecks, Alert
import random


class Command(BaseCommand):
    help = 'Set up demo mode and create sample intrusion data for testing'

    def add_arguments(self, parser):
        parser.add_argument(
            '--camera-id',
            type=int,
            help='Camera ID to create sample data for (default: first active camera)',
        )
        parser.add_argument(
            '--days',
            type=int,
            default=7,
            help='Number of days of sample data to create (default: 7)',
        )
        parser.add_argument(
            '--enable-demo',
            action='store_true',
            help='Enable demo mode',
        )

    def handle(self, *args, **options):
        self.stdout.write('Setting up intrusion detection demo...')
        
        # Enable demo mode if requested
        if options['enable_demo']:
            demo_config, created = SystemConfig.objects.get_or_create(
                key='demo_mode',
                defaults={
                    'value': 'true',
                    'category': 'demo',
                    'description': 'Enable demo mode with sample data'
                }
            )
            if not created:
                demo_config.value = 'true'
                demo_config.save()
            self.stdout.write(self.style.SUCCESS('✓ Demo mode enabled'))
        
        # Get camera
        camera_id = options.get('camera_id')
        if camera_id:
            try:
                camera = Camera.objects.get(id=camera_id)
            except Camera.DoesNotExist:
                self.stdout.write(self.style.ERROR(f'Camera with ID {camera_id} not found'))
                return
        else:
            camera = Camera.objects.filter(is_active=True).first()
            if not camera:
                self.stdout.write(self.style.ERROR('No active cameras found'))
                return
        
        self.stdout.write(f'Using camera: {camera.name} (ID: {camera.id})')
        
        # Create sample intrusion data
        days = options['days']
        now = timezone.now()
        
        # Delete existing sample data for this camera
        existing_count = Detection.objects.filter(
            camera=camera,
            detection_type='intrusion',
            description__icontains='Sample intrusion'
        ).count()
        
        if existing_count > 0:
            Detection.objects.filter(
                camera=camera,
                detection_type='intrusion',
                description__icontains='Sample intrusion'
            ).delete()
            self.stdout.write(f'Removed {existing_count} existing sample intrusions')
        
        # Create sample data over the specified number of days
        intrusion_types = ['unauthorized_entry', 'restricted_area', 'after_hours', 'perimeter_breach']
        severities = ['low', 'medium', 'high', 'critical']
        
        total_created = 0
        
        for day in range(days):
            day_start = now - timedelta(days=day)
            
            # Create 2-8 incidents per day with realistic patterns
            incidents_per_day = random.randint(2, 8)
            
            for incident in range(incidents_per_day):
                # Random time during the day (more likely during evening/night)
                if random.random() < 0.6:  # 60% chance of evening/night incident
                    hour = random.randint(18, 23) if random.random() < 0.7 else random.randint(0, 6)
                else:
                    hour = random.randint(7, 17)
                
                minute = random.randint(0, 59)
                
                incident_time = day_start.replace(
                    hour=hour,
                    minute=minute,
                    second=random.randint(0, 59),
                    microsecond=0
                )
                
                # Create detection record
                intrusion_type = random.choice(intrusion_types)
                severity = random.choice(severities)
                object_count = random.randint(1, 3)
                
                detection = Detection.objects.create(
                    camera=camera,
                    frameid=f"DEMO_{camera.id}_{incident_time.strftime('%Y%m%d_%H%M%S')}",
                    description=f"Sample intrusion detection - {intrusion_type.replace('_', ' ').title()} - {object_count} person(s) detected",
                    detection_type='intrusion',
                    object_count=object_count,
                    annotated_snapshot_path=f'demo/intrusion_sample_{total_created + 1}.jpg',
                    timestamp=incident_time
                )
                
                # Create intrusion check record
                intrusionChecks.objects.create(
                    camera=camera,
                    Detectionid=detection,
                    trackid=f"TRACK_{random.randint(1000, 9999)}",
                    intrusion=True,
                    intrusion_type=intrusion_type,
                    severity=severity,
                    details={
                        'demo_data': True,
                        'detection_confidence': round(random.uniform(0.7, 0.95), 2),
                        'roi_area': 'restricted_zone',
                        'timestamp': incident_time.isoformat()
                    }
                )
                
                # Create alert (some acknowledged, some not)
                Alert.objects.create(
                    camera=camera,
                    detection=detection,
                    severity=severity,
                    message=f"Security Intrusion Alert: {intrusion_type.replace('_', ' ').title()} detected",
                    is_acknowledged=random.random() < 0.7  # 70% chance of being acknowledged
                )
                
                total_created += 1
        
        self.stdout.write(
            self.style.SUCCESS(
                f'✓ Created {total_created} sample intrusion incidents over {days} days'
            )
        )
        
        # Show summary
        self.stdout.write('\n--- Summary ---')
        self.stdout.write(f'Camera: {camera.name} (ID: {camera.id})')
        self.stdout.write(f'Sample data period: {days} days')
        self.stdout.write(f'Total incidents created: {total_created}')
        
        # Count by severity
        for severity in severities:
            count = intrusionChecks.objects.filter(
                camera=camera,
                severity=severity,
                details__demo_data=True
            ).count()
            self.stdout.write(f'{severity.title()} severity: {count} incidents')
        
        self.stdout.write(
            self.style.SUCCESS('\n✓ Demo setup complete! Visit the intrusion detection page to see the data.')
        )