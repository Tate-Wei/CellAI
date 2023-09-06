"""
URL configuration for ami project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

import home.views as home_views
import main_page.views as mp_views

urlpatterns = [
    path('', home_views.home, name='home'),
    path('upload/', home_views.upload_file, name='upload_file'),
    path('training/', home_views.training, name='training'),
    path('prediction/', home_views.prediction, name='prediction'),
    path('main_page/', mp_views.main_page),
    path('about/', mp_views.about),
    path('settings/', mp_views.settings),
    path('save/', mp_views.save),
    path('admin/', admin.site.urls),
]
