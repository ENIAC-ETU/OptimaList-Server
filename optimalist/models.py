from django.db import models


class Product(models.Model):
    title = models.CharField(max_length=200)
    day = models.IntegerField()
    modified_date = models.DateTimeField(auto_now_add=True)
    interval = models.IntegerField(default=0)