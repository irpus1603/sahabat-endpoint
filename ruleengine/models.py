from django.db import models
from django.utils import timezone



class TimeStampedModel(models.Model):
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

class AIRuleSet(TimeStampedModel):
    """
    A named bundle of rules. Attach the same set to multiple jobs.
    """
    name        = models.CharField(max_length=120, unique=True)
    description = models.TextField(blank=True, null=True)
    enabled     = models.BooleanField(default=True)

    class Meta:
        db_table = "ai_rule_set"

    def __str__(self):
        return self.name


class AIRule(TimeStampedModel):
    """
    Rule compatible with your in-process rules engine (when_all/when_any/actions).
    """
    rule_set  = models.ForeignKey(AIRuleSet, on_delete=models.CASCADE, related_name="rules", db_index=True)
    name      = models.CharField(max_length=120)
    enabled   = models.BooleanField(default=True)
    priority  = models.IntegerField(default=100, db_index=True)
    when_all  = models.JSONField(default=list)  # [{path/op/value} or {fn/args}]
    when_any  = models.JSONField(default=list, blank=True, null=True)
    actions   = models.JSONField(default=list)  # [{type: "create_incident"|"email"|"webhook"|... , ...}]

class AIROI(TimeStampedModel):
    """
    Generic ROI geometry for a job. Keep it JSON to avoid heavy GIS stacks.
    """
    class ROIType(models.TextChoices):
        POLYGON = "POLYGON", "Polygon"
        LINE    = "LINE",    "Line"
        RECT    = "RECT",    "Rectangle"
        CIRCLE  = "CIRCLE",  "Circle"

    name        = models.CharField(max_length=120)
    roi_type    = models.CharField(max_length=12, choices=ROIType.choices)
    geometry    = models.JSONField(default=dict)  # e.g., {"points":[[x,y],...]} or {"p1":[x,y],"p2":[x,y]}
    metadata    = models.JSONField(default=dict, blank=True, null=True)

    class Meta:
        db_table = "ai_roi"

    def __str__(self):
        return f"{self.name} ({self.roi_type})"

