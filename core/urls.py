
from django.urls import path
from .views import *


urlpatterns = [
    path('', FileView.as_view(), name="Images")
]