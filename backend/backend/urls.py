
from django.contrib import admin
from django.urls import path, include
from main.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('placeorder', main),
    path('calories', cals),
    path('exercise', exercise),
    path('ingredients', ingredient),
    path('recepies', recepies),
    path('addfood', addfoods),
    path('home', home),
    path('', include('auth0login.urls'))
]
