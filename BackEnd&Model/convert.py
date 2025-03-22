from tensorflow.keras.models import load_model

# Load an existing Keras model (saved in .h5 format or SavedModel format)
model = load_model('best_face_recognition_model.keras')  # Replace 'model.h5' with your file path

model_json = model.to_json()

with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model_weights.h5')

