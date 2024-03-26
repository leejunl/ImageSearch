
from django.urls import path,re_path

from . import views

app_name="chy"

urlpatterns = [
    path('upload/',views.image_search_view),    
    path('spyder/',views.Spyder),    
]
