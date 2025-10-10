from django.core.management.base import BaseCommand
from django.contrib.auth.models import Group, Permission, User
from django.contrib.contenttypes.models import ContentType
from core.models import Camera

class Command(BaseCommand):
    help = 'Create user groups and assign permissions for SOCA Edge access control'

    def handle(self, *args, **options):
        self.stdout.write('Setting up user groups and permissions...')
        
        # Create groups
        admin_group, admin_created = Group.objects.get_or_create(name='admin')
        users_group, users_created = Group.objects.get_or_create(name='users')
        
        if admin_created:
            self.stdout.write(self.style.SUCCESS('Created "admin" group'))
        else:
            self.stdout.write('Admin group already exists')
            
        if users_created:
            self.stdout.write(self.style.SUCCESS('Created "users" group'))
        else:
            self.stdout.write('Users group already exists')
        
        # Get Camera model permissions
        try:
            content_type = ContentType.objects.get_for_model(Camera)
            camera_permissions = Permission.objects.filter(content_type=content_type)
            
            self.stdout.write(f'Found {camera_permissions.count()} camera permissions')
            
            # Admin group gets all Camera permissions
            admin_group.permissions.set(camera_permissions)
            self.stdout.write(self.style.SUCCESS('Assigned all camera permissions to admin group'))
            
            # Users group gets only view permission
            view_camera_perm = Permission.objects.get(
                codename='view_camera', 
                content_type=content_type
            )
            users_group.permissions.set([view_camera_perm])
            self.stdout.write(self.style.SUCCESS('Assigned view-only permission to users group'))
            
        except Permission.DoesNotExist:
            self.stdout.write(
                self.style.ERROR('Camera permissions not found. Make sure migrations are run.')
            )
            return
        
        # Assign existing superusers to admin group
        superusers = User.objects.filter(is_superuser=True)
        for user in superusers:
            user.groups.add(admin_group)
            self.stdout.write(f'Added superuser "{user.username}" to admin group')
        
        # Show current group assignments
        self.stdout.write('\n--- Current Group Permissions ---')
        self.stdout.write(f'Admin group permissions: {admin_group.permissions.count()}')
        for perm in admin_group.permissions.all():
            self.stdout.write(f'  - {perm.codename}')
            
        self.stdout.write(f'Users group permissions: {users_group.permissions.count()}')
        for perm in users_group.permissions.all():
            self.stdout.write(f'  - {perm.codename}')
            
        self.stdout.write('\n--- Users in Groups ---')
        admin_users = User.objects.filter(groups=admin_group)
        users_users = User.objects.filter(groups=users_group)
        
        self.stdout.write(f'Admin group members: {admin_users.count()}')
        for user in admin_users:
            self.stdout.write(f'  - {user.username}')
            
        self.stdout.write(f'Users group members: {users_users.count()}')
        for user in users_users:
            self.stdout.write(f'  - {user.username}')
        
        self.stdout.write(self.style.SUCCESS('\nGroups and permissions setup completed successfully!'))
        self.stdout.write('You can now assign users to groups via Django admin or by running:')
        self.stdout.write('  python manage.py shell')
        self.stdout.write('  >>> from django.contrib.auth.models import User, Group')
        self.stdout.write('  >>> user = User.objects.get(username="username")')
        self.stdout.write('  >>> group = Group.objects.get(name="users")')
        self.stdout.write('  >>> user.groups.add(group)')