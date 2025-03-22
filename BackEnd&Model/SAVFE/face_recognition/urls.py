from django.urls import path
from .views import FaceRecognitionView

#In this part, a URL (...)/predict/ is created, which functions from the FaceRecognitionView 
#object that has been imported from face_recognition.view.
#The (...) represents the URL of the SAVFE project.
urlpatterns = [
    path('predict/', FaceRecognitionView.as_view(), name='face_recognition_predict'),
]
