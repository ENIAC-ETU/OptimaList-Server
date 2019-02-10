from django.contrib import admin
from django.urls import include, path
from optimalist import views

urlpatterns = [
    path('get-ocr/', views.get_ocr)
]
