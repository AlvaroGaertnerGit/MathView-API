from django.contrib import admin
from django.urls import path, re_path,include;
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

schema_view = get_schema_view(
   openapi.Info(
      title="API de Funciones Complejas",
      default_version='v1',
      description="Calcula funciones complejas y devuelve la magnitud y fase",
      terms_of_service="https://www.google.com/policies/terms/",
      contact=openapi.Contact(email="contact@myapi.com"),
      license=openapi.License(name="BSD License"),
   ),
   public=True,
   permission_classes=(permissions.AllowAny,),
)
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('Calcs.urls')),
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0),
         name='swagger-docs'),
    
]
