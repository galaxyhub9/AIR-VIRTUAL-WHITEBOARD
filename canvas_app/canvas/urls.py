from django.urls import path
from . import views

app_name = "canvasapp"

urlpatterns = [
    path('', views.canvas_view, name='canvas'),
    path('run_canvas/', views.run_canvas, name='run_canvas'),
]
