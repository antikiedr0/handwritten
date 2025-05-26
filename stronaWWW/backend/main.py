import base64
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Wczytaj model (z Twojego skryptu treningowego)
model = tf.keras.models.load_model('handwritten.keras')

def preprocess_image(img_b64):
    # Usuń nagłówek base64
    header, encoded = img_b64.split(',', 1)
    img_bytes = base64.b64decode(encoded)

    # Otwórz obraz i konwertuj do grayscale
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    img = img.resize((28, 28))

    img_array = np.array(img).astype(np.float32)

    # Odwróć kolory: jeśli masz czarną cyfrę na białym tle, MNIST ma odwrotnie
    img_array = 255 - img_array

    # Normalizacja tak jak w tf.keras.utils.normalize z axis=1:
    # to normalizacja wzdłuż osi 1, więc wykonujemy ręcznie:
    # normalizacja polega na dzieleniu każdej próbki przez normę L2 wzdłuż osi 1

    # Obliczamy normę L2 po kolumnach (axis=1)
    norms = np.linalg.norm(img_array, ord=2, axis=1, keepdims=True)
    # Zapobiegamy dzieleniu przez zero
    norms[norms == 0] = 1

    img_norm = img_array / norms

    # Dodaj wymiar batch size (1, 28, 28)
    img_norm = np.expand_dims(img_norm, axis=0)

    return img_norm

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        img_array = preprocess_image(data['image'])
        prediction = model.predict(img_array)
        predicted_digit = int(np.argmax(prediction))

        return jsonify({'prediction': predicted_digit})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
