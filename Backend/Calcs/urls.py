from django.urls import path
from . import views  

urlpatterns = [
    path('calculateFunction/', views.calculateFunction, name='calculateFunction'), 
    path('calculateFunctionParam/', views.calculateFunctionParam, name='calculateFunctionParam'), 
    path('calculateZeta/', views.calculateZeta, name='calculateZeta'), 
    path('calculateFourier/', views.calculateFourier, name='calculateFourier'), 
    path('calculateMandelbrot/', views.calculateMandelbrot, name='calculateMandelbrot'), 
]