from django.urls import path
from .views import chatbot_response, chat

urlpatterns = [
    path('response/', chatbot_response, name='chatbot_response'),
    path('', chat, name='home'),
]
