from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Homepage
    path('crop-recommend/', views.crop_recommend, name='crop_recommend'),
    path('yield-predict/', views.yield_predict, name='yield_predict'),
]
