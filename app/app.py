import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gdown
import zipfile
import torch
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from flask import Flask, request, render_template
from PIL import Image
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
from safetensors.torch import load_file
from io import BytesIO

# from flask import Flask, request, render_template
# from PIL import Image
# from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
# from safetensors.torch import load_file
# from io import BytesIO

# # file_url = 'https://drive.google.com/file/d/1IsNhGoa8O6_XGe48IK5a-oHBY198C9bO/view?usp=sharing'
# # downloaded_file = 'Project_inference.zip'
# # gdown.download(file_url, downloaded_file, fuzzy=True)
# # extract_dir = '.'
# # os.makedirs(extract_dir, exist_ok=True)

app = Flask(__name__)

model_dir = os.path.abspath("./Project_inference/effNetb5_originpreprocessor_rt15")

# Load the preprocessor
preprocessor = EfficientNetImageProcessor.from_pretrained(model_dir)

# Load the model config
config_path = os.path.join(model_dir, "config.json")
config = EfficientNetForImageClassification.config_class.from_pretrained(config_path)

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
            label = model.config.id2label.get(predicted_label, 'Unknown stage')
        
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Plot probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        probabilities_percent = probabilities * 100
        labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        
        plt.figure(figsize=(10, 5))
        bars = plt.bar(range(len(probabilities_percent)), probabilities_percent, color='orange')
        plt.xticks(range(len(probabilities_percent)), labels, rotation=0, fontsize = 18,  ha='center', color='white')
        plt.yticks(fontsize=18, color='white')
        plt.ylabel('Probability, %', fontsize=18, color='white')
        plt.gca().set_facecolor('black')
        plt.gcf().set_facecolor('black')
        plt.tick_params(colors='white')

        for bar, prob in zip(bars, probabilities_percent):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{prob:.1f}%", ha='center', va='bottom', color='white', fontsize=12)

        plt.tight_layout()
        plot_buffer = BytesIO()
        plt.savefig(plot_buffer, format='png', bbox_inches='tight')
        plt.close()
        plot_data = base64.b64encode(plot_buffer.getvalue()).decode()

        return render_template('index.html', label=label, image_data=img_str, plot_data=plot_data)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8893, debug=True)
