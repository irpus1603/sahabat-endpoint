# Generated manually for camera_source field

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0006_camera_source'),
    ]

    operations = [
        migrations.AddField(
            model_name='camera',
            name='camera_source',
            field=models.CharField(
                choices=[
                    ('local', 'Local Camera'),
                    ('rtsp', 'RTSP Stream'),
                    ('http', 'HTTP Stream'),
                    ('video_file', 'Video File'),
                    ('youtube', 'YouTube URL'),
                ],
                default='rtsp',
                help_text='Type of camera source',
                max_length=20
            ),
        ),
    ]