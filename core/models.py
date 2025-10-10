import json
import numpy as np
import cv2
from hashlib import blake2b
from typing import Annotated, Tuple
from django.db import models
from django.utils.choices import BlankChoiceIterator

# class for camera object
class Camera(models.Model):
    source_choices = [
        ('local', 'Local Camera'),
        ('rtsp', 'RTSP Stream'),
        ('http', 'HTTP Stream'),
        ('video_file', 'Video File'),
        ('youtube', 'YouTube URL'),
    ]

    roi_shape_choices = [
        ('rectangle', 'Rectangle'),
        ('polygon', 'Polygon'),
        ('line', 'Line'),
        ('circle', 'Circle'),
    ]

    name = models.CharField(max_length=100, blank=True, null=True)
    site_name = models.CharField(max_length=100, blank=True, null=True)
    floor = models.CharField(max_length=100, blank=True, null=True)
    location = models.CharField(max_length=255, blank=True, null=True)
    thumbnail = models.ImageField(upload_to='thumbnails/', blank=True, null=True)
    rtsp_url = models.TextField(blank=True, null=True)
    username = models.CharField(max_length=100, blank=True, null=True)
    password = models.CharField(max_length=100, blank=True, null=True)
    camera_source = models.CharField(max_length=20, choices=source_choices, default='rtsp', help_text="Type of camera source")
    is_active = models.BooleanField(default=True)
    roi_coordinates = models.TextField(blank=True, null=True, help_text="Region of Interest coordinates in JSON format")
    roi_type = models.CharField(max_length=20, choices=roi_shape_choices, default='polygon', blank=True, null=True, help_text="Type of ROI shape")
    aimodule= models.ForeignKey('Aimodule', on_delete=models.SET_NULL, null=True, blank=True)
    is_aimodule_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add = True)

    def __str__(self):
        return f"Camera {self.id}: {self.name}"
    
    def get_roi_coordinates(self):
        """Returns ROI data in appropriate format based on ROI type"""
        if not self.roi_coordinates:
            return None
        
        # Handle both string (JSON) and dict formats
        if isinstance(self.roi_coordinates, str):
            try:
                coords = json.loads(self.roi_coordinates)
            except json.JSONDecodeError:
                return None
        elif isinstance(self.roi_coordinates, (list, dict)):
            coords = self.roi_coordinates
        else:
            return None
        
        # If coords is a list (multiple ROI regions), process each one
        if isinstance(coords, list) and len(coords) > 0:
            processed_regions = []
            for region in coords:
                if isinstance(region, dict) and 'type' in region:
                    processed_region = self._process_roi_region(region)
                    if processed_region:
                        processed_regions.append(processed_region)
            return processed_regions if processed_regions else None
        elif isinstance(coords, dict) and 'type' in coords:
            # Single ROI region
            return self._process_roi_region(coords)
        
        # Legacy format - assume polygon
        return self._process_legacy_coordinates(coords)
    
    def _process_roi_region(self, region):
        """Process a single ROI region based on its type"""
        roi_type = region.get('type')
        
        if roi_type == 'polygon':
            points = region.get('points', [])
            if not points:
                return None
            flattened_coords = []
            for point in points:
                if isinstance(point, dict) and 'x' in point and 'y' in point:
                    try:
                        x, y = float(point['x']), float(point['y'])
                        flattened_coords.append([int(x), int(y)])
                    except (ValueError, TypeError):
                        continue
            if flattened_coords:
                return {
                    'type': 'polygon',
                    'points': np.array(flattened_coords, dtype=np.int32).reshape((-1, 1, 2))
                }
        
        elif roi_type == 'rectangle':
            try:
                return {
                    'type': 'rectangle',
                    'x': int(region.get('x', 0)),
                    'y': int(region.get('y', 0)),
                    'width': int(region.get('width', 0)),
                    'height': int(region.get('height', 0))
                }
            except (ValueError, TypeError):
                return None
        
        elif roi_type == 'line':
            try:
                return {
                    'type': 'line',
                    'x1': int(region.get('x1', 0)),
                    'y1': int(region.get('y1', 0)),
                    'x2': int(region.get('x2', 0)),
                    'y2': int(region.get('y2', 0))
                }
            except (ValueError, TypeError):
                return None
        
        elif roi_type == 'circle':
            try:
                return {
                    'type': 'circle',
                    'centerX': int(region.get('centerX', 0)),
                    'centerY': int(region.get('centerY', 0)),
                    'radius': int(region.get('radius', 0))
                }
            except (ValueError, TypeError):
                return None
        
        return None
    
    def _process_legacy_coordinates(self, coords):
        """Process legacy coordinate format (for backward compatibility)"""
        # Handle legacy polygon format
        if isinstance(coords, dict):
            if 'coordinates' in coords:
                coords = coords['coordinates']
            elif 'points' in coords:
                coords = coords['points']
        
        if isinstance(coords, list) and len(coords) > 0:
            flattened_coords = []
            for item in coords:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    try:
                        x, y = float(item[0]), float(item[1])
                        flattened_coords.append([int(x), int(y)])
                    except (ValueError, TypeError):
                        continue
                elif isinstance(item, dict) and 'x' in item and 'y' in item:
                    try:
                        x, y = float(item['x']), float(item['y'])
                        flattened_coords.append([int(x), int(y)])
                    except (ValueError, TypeError):
                        continue
            
            if flattened_coords:
                return {
                    'type': 'polygon',
                    'points': np.array(flattened_coords, dtype=np.int32).reshape((-1, 1, 2))
                }
        
        return None

    def set_roi_coordinates(self, coordinates):
        """Expects list of [x, y] points"""
        self.roi_coordinates = json.dumps(coordinates)
    
    def get_roi_polygon(self):
        """Returns ROI polygon data for polygon and rectangle types"""
        roi_data = self.get_roi_coordinates()
        if not roi_data:
            return None
        
        # Handle multiple ROI regions
        if isinstance(roi_data, list):
            polygons = []
            for region in roi_data:
                if isinstance(region, dict):
                    if region.get('type') == 'polygon':
                        polygons.append(region['points'])
                    elif region.get('type') == 'rectangle':
                        # Convert rectangle to polygon
                        x, y, w, h = region['x'], region['y'], region['width'], region['height']
                        rect_points = np.array([
                            [x, y], [x + w, y], [x + w, y + h], [x, y + h]
                        ], dtype=np.int32).reshape((-1, 1, 2))
                        polygons.append(rect_points)
            return polygons[0] if len(polygons) == 1 else (polygons if polygons else None)
        
        # Handle single ROI region
        elif isinstance(roi_data, dict):
            if roi_data.get('type') == 'polygon':
                return roi_data['points']
            elif roi_data.get('type') == 'rectangle':
                # Convert rectangle to polygon
                x, y, w, h = roi_data['x'], roi_data['y'], roi_data['width'], roi_data['height']
                return np.array([
                    [x, y], [x + w, y], [x + w, y + h], [x, y + h]
                ], dtype=np.int32).reshape((-1, 1, 2))
        
        return None

class Detection(models.Model):
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    frameid = models.CharField(max_length=100, blank=True, null=True)
    detection_type = models.CharField(max_length=255, blank=True, null=True)
    object_count = models.IntegerField(default=1, blank=True, null=True)
    description = models.TextField(blank=True, null = True)
    annotated_snapshot_path = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['camera','-timestamp'])
        ]

    def __str__(self):
        return f"Detection {self.id} - Camera {self.camera.name}"

class PPEChecks(models.Model):
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    Detectionid = models.ForeignKey(Detection, on_delete=models.CASCADE, null=True, blank=True)
    trackid = models.CharField(max_length=100, blank=True, null=True)
    has_helmet = models.BooleanField(default=False)
    has_vest   = models.BooleanField(default=False) 
    has_goggles= models.BooleanField(default=False) 
    has_gloves = models.BooleanField(default=False)
    has_shoes = models.BooleanField(default=False)
    details = models.JSONField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(fields=['camera','-Detectionid','trackid'])
        ]

    def __str__(self):
        return f"PPE Check {self.id} - Camera {self.camera.name}"   

class intrusionChecks(models.Model):
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    Detectionid = models.ForeignKey(Detection, on_delete=models.CASCADE, null=True, blank=True)
    trackid = models.CharField(max_length=100, blank=True, null=True)
    intrusion = models.BooleanField(default=False)
    intrusion_type = models.CharField(max_length=255, blank=True, null=True)
    severity = models.CharField(max_length=50, blank=True, null=True)
    details = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)


    class Meta:
        indexes = [
            models.Index(fields=['camera','-Detectionid','trackid'])
        ]

    def __str__(self):
        return f"Intrusion Check {self.id} - Camera {self.camera.name}"

class PeopleCountingEvent(models.Model):
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    frameid = models.CharField(max_length=100, blank=True, null=True)

    person_id = models.CharField(max_length=100, blank=True, null=True)
    direction = models.CharField(max_length=50, choices=[("IN", "In"), ("OUT", "Out")])
    cumulative_in = models.IntegerField(default=0)
    cumulative_out = models.IntegerField(default=0)
    occupancy = models.IntegerField(default=0)

    snapshot_path = models.CharField(max_length=255, blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['camera','-timestamp']),
            models.Index(fields=['direction']),
        ]

    def __str__(self):
        return f"PeopleCounting {self.id} - {self.direction} - {self.camera.name}"


class Alert(models.Model):
    SEVERITY_CHOICES = [
        ('low','low'),
        ('medium','medium'),
        ('high','high'),
        ('critical','critical'),
    ]

    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    detection = models.ForeignKey(Detection, on_delete=models.SET_NULL, null=True)
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES)
    message = models.TextField()
    is_acknowledged = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add = True)


class SystemConfig(models.Model):
    key = models.CharField(max_length=100, unique = True)
    value = models.TextField()
    category = models.CharField(max_length=50, default='general')
    description = models.TextField()

    @classmethod
    def get_value(cls, key, default=None):
        try:
            return cls.objects.get(key=key).value
        except cls.DoesNotExist:
            return default
    
class Aimodule(models.Model):
    name=models.CharField(unique=True, max_length=200, blank=False)
    module=models.CharField(max_length=100, blank=False)
    description=models.TextField(blank=True, null=True)
    is_active=models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
