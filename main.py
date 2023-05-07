from flask import Flask, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
import io
import json
from PIL import Image

app = Flask(__name__)
CORS(app)
model = tf.saved_model.load("models/v1")

quality = {}

@app.route('/predict', methods=['POST'])
def predict(): 
    input_data = request.get_json()

    encoded_image = input_data['image']
    image_data = base64.b64decode(encoded_image)

    image = Image.open(io.BytesIO(image_data))
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0) 
    input_tensor = tf.convert_to_tensor(image)

    output_tensor = model(input_tensor)

    prediction = np.array(output_tensor)[0]

    quality["good"] = prediction[0]
    quality["normal"] = prediction[1]
    quality["bad"] = prediction[2]

    return max(zip(quality.values(), quality.keys()))[1]

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)