"""
Django management command to start the SOCA Edge video processor
Usage: python manage.py start_video_processor
"""

from django.core.management.base import BaseCommand
from django.conf import settings
import sys
import os
from pathlib import Path

class Command(BaseCommand):
    help = 'Start the SOCA Edge video processor for surveillance cameras'

    def add_arguments(self, parser):
        parser.add_argument(
            '--cleanup-only',
            action='store_true',
            help='Run cleanup only without starting video processing',
        )
        parser.add_argument(
            '--max-streams',
            type=int,
            default=8,
            help='Maximum number of concurrent camera streams (default: 8)',
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('Starting SOCA Edge Video Processor...')
        )

        try:
            # Import video processor
            from video_processor.video_processor_standalone import VideoProcessor
            
            processor = VideoProcessor()
            
            if options['cleanup_only']:
                self.stdout.write('Running cleanup only...')
                processor.cleanup_old_data()
                self.stdout.write(
                    self.style.SUCCESS('Cleanup completed successfully')
                )
                return
            
            # Override max streams if specified
            if options['max_streams']:
                processor.max_concurrent_streams = options['max_streams']
                self.stdout.write(f'Max concurrent streams set to: {options["max_streams"]}')
            
            # Start processing
            if processor.start_processing():
                self.stdout.write(
                    self.style.SUCCESS(
                        f'Video processor started with {len(processor.processing_threads)} camera streams'
                    )
                )
                self.stdout.write('Press Ctrl+C to stop the processor')
                
                try:
                    # Keep the command running
                    import time
                    last_cleanup = time.time()
                    
                    while True:
                        time.sleep(60)  # Check every minute
                        
                        # Perform cleanup every hour
                        if time.time() - last_cleanup > 3600:
                            processor.cleanup_old_data()
                            last_cleanup = time.time()
                            self.stdout.write('Hourly cleanup completed')
                            
                except KeyboardInterrupt:
                    self.stdout.write('\nReceived interrupt signal, stopping...')
                finally:
                    processor.stop_processing()
                    self.stdout.write(
                        self.style.SUCCESS('Video processor stopped successfully')
                    )
            else:
                self.stdout.write(
                    self.style.ERROR('Failed to start video processor')
                )
                
        except ImportError as e:
            self.stdout.write(
                self.style.ERROR(f'Failed to import video processor: {e}')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error starting video processor: {e}')
            )