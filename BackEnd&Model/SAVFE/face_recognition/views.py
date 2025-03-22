import tensorflow as tf
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageUploadSerializer
from PIL import Image
import numpy as np
import io

#In this section, we set the model with the directory of our own model
model = tf.keras.models.load_model(r'C:/Users/LENOVO/Latihan AoL AI/best_face_recognition_model.keras')


class FaceRecognitionView(APIView):
    def post(self, request, *args, **kwargs):

        
        #This serializer code is used to send the data received from the frontend to be validated in the ImageUploadSerializer.
        #The data=request.data part means initializing the data from request.data, which is the data obtained from the frontend.
        serializer = ImageUploadSerializer(data=request.data)

        #I the serializer is valid then the posted data from the frontend is a image
        if serializer.is_valid():

            #In this part of the code, it will perform validation again if the data is an image.
            #Then, if it is validated, the image will be opened.
            image_file = serializer.validated_data['image']
            image = Image.open(image_file)

            #In this section the image will be resized to 299x299 pixels
            image = image.resize((299, 299))

            #In this section the image will be convert into numeric array
            image_array = np.array(image) / 255.0

            #In this section we add the image dimension from the numeric array
            image_array = np.expand_dims(image_array, axis=0) 

            #In this section, the model will do the prediction using our model
            predictions = model.predict(image_array)

            #In this section we will take the index of class with the highest confidence 
            predicted_class = np.argmax(predictions, axis=1)

            #In this section we will take the highest confidence score
            confidence = np.max(predictions)

            #In this section, we create an array with all of our classes name
            class_names = ['Jennifer Lawrence', 'Johnny Depp', 'Leonardo Dicaprio', 'Megan Fox', 'Michael Wijaya', 'Robert Dawney Jr', 'Scarlet Johansson', 'Tom Cruise', 'Tom Hanks', 'Will Smith']  # Add the actual names here

            #In this section we take the name of the highest predicted class
            predicted_person = class_names[predicted_class[0]]

            #Return the response of the predicted person, and the confidence score to the frontend
            return Response({
                'predicted_person': predicted_person,
                'confidence': float(confidence)
            }, status=status.HTTP_200_OK)

        #If the validation failed, the response will return the error to the fronted
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
