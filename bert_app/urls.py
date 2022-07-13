from django.urls import path
from bert_app import views
urlpatterns = [
    path('dic', views.trainToDev.as_view()),
    path('dic/', views.trainToDev.as_view()),
    path('question', views.bert_Question.as_view()),
    path('question/', views.bert_Question.as_view()), 
] 