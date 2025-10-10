"""
Template tags for authentication and permission checking in SOCA Edge
"""
from django import template
from django.contrib.auth.models import Group
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter(name='has_group')
def has_group(user, group_name):
    """
    Check if user belongs to a specific group
    Usage: {% if user|has_group:"admin" %}
    """
    if not user.is_authenticated:
        return False
    try:
        group = Group.objects.get(name=group_name)
        return group in user.groups.all()
    except Group.DoesNotExist:
        return False

@register.filter(name='is_admin')
def is_admin(user):
    """
    Check if user is in admin group
    Usage: {% if user|is_admin %}
    """
    if not user.is_authenticated:
        return False
    return user.groups.filter(name='admin').exists()

@register.filter(name='is_user')
def is_user(user):
    """
    Check if user is in users group
    Usage: {% if user|is_user %}
    """
    if not user.is_authenticated:
        return False
    return user.groups.filter(name='users').exists()

@register.filter(name='can_delete_camera')
def can_delete_camera(user):
    """
    Check if user can delete cameras
    Usage: {% if user|can_delete_camera %}
    """
    if not user.is_authenticated:
        return False
    return user.has_perm('core.delete_camera')

@register.filter(name='can_add_camera')
def can_add_camera(user):
    """
    Check if user can add cameras
    Usage: {% if user|can_add_camera %}
    """
    if not user.is_authenticated:
        return False
    return user.has_perm('core.add_camera')

@register.filter(name='can_change_camera')
def can_change_camera(user):
    """
    Check if user can edit/change cameras
    Usage: {% if user|can_change_camera %}
    """
    if not user.is_authenticated:
        return False
    return user.has_perm('core.change_camera')

@register.filter(name='can_view_camera')
def can_view_camera(user):
    """
    Check if user can view cameras
    Usage: {% if user|can_view_camera %}
    """
    if not user.is_authenticated:
        return False
    return user.has_perm('core.view_camera')

@register.simple_tag
def user_role_badge(user):
    """
    Return a badge HTML for user role
    Usage: {% user_role_badge user %}
    """
    if not user.is_authenticated:
        return ''
    
    if user.groups.filter(name='admin').exists():
        return mark_safe('<span class="badge bg-danger ms-1">Admin</span>')
    elif user.groups.filter(name='users').exists():
        return mark_safe('<span class="badge bg-info ms-1">User</span>')
    else:
        return mark_safe('<span class="badge bg-secondary ms-1">No Group</span>')

@register.simple_tag
def user_role_text(user):
    """
    Return user role as text
    Usage: {% user_role_text user %}
    """
    if not user.is_authenticated:
        return 'Guest'
    
    if user.groups.filter(name='admin').exists():
        return 'Admin'
    elif user.groups.filter(name='users').exists():
        return 'User'
    else:
        return 'No Group'