"""
People counting module for SOCA Edge system.

This module provides a modular and organized approach to people counting
using YOLO detection, video stream processing, and database management.

Main components:
- utils: Utility functions for frame processing and drawing
- video_capture: Video source management and monitoring
- model_manager: YOLO model management and configuration
- processor: Main people counting logic and tracking
- database_operations: Database operations for events and snapshots
"""

from .people_counting import start_detection_stream, send_telegram_async, save_snapshot_with_intrusion_async

__all__ = [
    'start_detection_stream',
    'send_telegram_async', 
    'save_snapshot_with_intrusion_async'
]