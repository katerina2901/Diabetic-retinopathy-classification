import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import gdown
import zipfile
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys

from flask import Flask, request, render_template
from PIL import Image
from transformers import AutoImageProcessor
from safetensors.torch import load_file
from io import BytesIO
from Validation import MedViTConfig, MedViTClassification, extract_attention_map, plot_attention_map

sys.path.append(os.path.abspath('./MedViT'))

print("If it's a first run, need to Download weights")
print('Download weights? Yes/No')
flag = str(input(''))
flag = True if flag == "Yes" else False
if flag:
    file_url = 'https://drive.google.com/file/d/1wyu2BQ6I96jBfa7CEOZcnG-DJpjSpaZN/view?usp=sharing'
    downloaded_file = 'Project_inference.zip'
    gdown.download(file_url, downloaded_file, fuzzy=True)
    with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
        zip_ref.extractall('.')

    print('Im here')

app = Flask(__name__)

model_dir = os.path.abspath("./Project_inference/MedViT512_tr35_stage6(3)_CCropSpot2HTrivAug_fastvitprepr_lr1e5")
# Image processor load
preprocessor = AutoImageProcessor.from_pretrained(model_dir)
preprocessor.size['height'] = 512
preprocessor.size['width'] = 512

# Load the model configuration and initialize the model
config = MedViTConfig.from_pretrained(os.path.join(model_dir, "config.json"))
model = MedViTClassification(config, pretrained=True)
model_weights = load_file(os.path.join(model_dir, "model.safetensors"))
model.load_state_dict(model_weights, strict=False)
model.eval()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

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
        inputs = preprocessor(image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs['logits']
            label = logits.argmax(-1).item()
            # Extracting the attention map
            attn_heatmap_resized = extract_attention_map(model, inputs['pixel_values'], device=device)

            # Visualizing an attention map
            attention_data = plot_attention_map(image, attn_heatmap_resized)

        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Plotting the probability graph
        probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        probabilities_percent = probabilities * 100
        labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        
        plt.figure(figsize=(10, 5))
        bars = plt.bar(range(len(probabilities_percent)), probabilities_percent, color='orange')
        plt.xticks(range(len(probabilities_percent)), labels, rotation=0, fontsize=18, ha='center', color='white')
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

        return render_template('index.html', label=label, image_data=img_str, plot_data=plot_data, attention_data=attention_data)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8893, use_reloader=False, debug=True)
