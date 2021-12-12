
from os import name
from django.contrib import admin
from django.urls import path, include
from main.views import *
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('placeorder', main),
    path('calories', cals),
    path('exercise', exercise),
    path('ingredients', ingredient),
    path('recepies', recepies),
    path('addfood', addfoods),
    path('home', home),
    path('', include('auth0login.urls')),
    path('', dashboard),
    path('chef', chefHome, name='chefHome'),
    path('orders', chef, name='chef'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
