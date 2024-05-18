import os
from flask import Flask, request, render_template
import torch
from PIL import Image
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
from safetensors.torch import load_file

app = Flask(__name__)

# Определите абсолютный путь к директории с моделью
model_dir = os.path.abspath("./saved_models/effNetb5_originpreprocessor_rt15")

# Load the preprocessor
preprocessor = EfficientNetImageProcessor.from_pretrained(model_dir)

# Load the model config
config = EfficientNetForImageClassification.from_pretrained(model_dir).config

# Initialize the model
model = EfficientNetForImageClassification(config)

# Load the model weights from safetensors
model_weights = load_file(os.path.join(model_dir, "model.safetensors"))
model.load_state_dict(model_weights)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        image = Image.open(file.stream)
        inputs = preprocessor(image, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_label = logits.argmax(-1).item()
        label = model.config.id2label[predicted_label]
        
        return render_template('index.html', label=label)

if __name__ == "__main__":
    app.run(debug=True)
