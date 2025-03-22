from rest_framework import serializers



#In this part, it is used to validate if the data sent from the frontend to the backend is in the form of a photo. 
#This is checked through the ImageField part.
class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()