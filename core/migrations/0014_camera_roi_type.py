# Generated manually for roi_type field addition

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0013_alter_camera_roi_coordinates'),
    ]

    operations = [
        migrations.AddField(
            model_name='camera',
            name='roi_type',
            field=models.CharField(
                blank=True,
                choices=[('rectangle', 'Rectangle'), ('polygon', 'Polygon'), ('line', 'Line'), ('circle', 'Circle')],
                default='polygon',
                help_text='Type of ROI shape',
                max_length=20,
                null=True
            ),
        ),
    ]