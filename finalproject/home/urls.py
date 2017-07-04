from django.conf.urls import url, include
from . import views

urlpatterns = [

    url(r'^$', views.home, name='home'),
    url(r'^sum_result/', views.result, name='result'),
    url(r'^summ1/', views.summarizer, name='summarizer'),
    url(r'^summ2/', views.summarizer1, name='summarizer1'),
    url(r'^result1/', views.result1, name='result1'),
]