from django.db import models


class Product(models.Model):
    title = models.CharField(max_length=200)
    day = models.IntegerField()
    modified_date = models.DateTimeField(auto_now_add=True)
    interval = models.IntegerField(default=0)


class StorageCredentials(models.Model):
    type = models.CharField(max_length=200)
    project_id = models.CharField(max_length=200)
    private_key_id = models.CharField(max_length=200)
    private_key = models.TextField(max_length=2000)
    client_email = models.CharField(max_length=200)
    client_id = models.CharField(max_length=200)
    auth_uri = models.CharField(max_length=200)
    token_uri = models.CharField(max_length=200)
    auth_provider_x509_cert_url = models.CharField(max_length=200)
    client_x509_cert_url = models.CharField(max_length=500)