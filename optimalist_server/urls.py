from django.contrib import admin
from django.urls import include, path
from optimalist import views

urlpatterns = [
    path('train-model/', views.train_model),
    path('get-prediction/', views.get_prediction),
    path('evaluate-model/', views.evaluate_model)
]
