
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import requests
import os
from datetime import datetime
from ultralytics import YOLO
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from serpapi import GoogleSearch

app = Flask(__name__, template_folder="templates", static_folder="static")
os.makedirs('static', exist_ok=True)

# === Cargar modelo entrenado ===
model = YOLO("../app/models/best.pt")
print("✅ Modelo cargado. Clases:", model.names)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict-url', methods=['POST'])
def predict_url():
    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return jsonify({'error': 'Image download failed'}), 400

        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        results = model.predict(img, conf=0.4)[0]
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        nombres_detectados = [model.names[cid].lower() for cid in class_ids]
        hay_plaga = any('maleza' in nombre for nombre in nombres_detectados)

        return jsonify({
            'result': 'Se detectó presencia de plaga o maleza' if hay_plaga else 'Planta saludable',
            'detections': nombres_detectados
        })

    except Exception as e:
        return jsonify({'error': 'Error al procesar la imagen', 'details': str(e)}), 500

@app.route('/labels', methods=['GET'])
def get_labels():
    return jsonify({'labels': model.names})

@app.route('/search', methods=['POST'])
def buscar_google():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({'results': []})

    params = {
        "engine": "google",
        "q": query,
        "tbm": "isch",
        "api_key": "3feebd4ff3a67a8e36b327e8fda8c798a059548ee7ae1c16413649b8c80c632e"
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        urls = [img['original'] for img in results.get('images_results', [])]
        return jsonify({'results': urls[:5]})
    except Exception as e:
        return jsonify({'error': 'Búsqueda fallida', 'details': str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió archivo'}), 400

    file = request.files['file']
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Error al leer el CSV: {str(e)}'}), 400

    if 'url' not in df.columns or 'label' not in df.columns:
        return jsonify({'error': 'El CSV debe tener columnas: url,label'}), 400

    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        url = row['url']
        true_label = int(row['label'])

        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue

            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            results = model.predict(img, conf=0.4)[0]
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            nombres_detectados = [model.names[cid].lower() for cid in class_ids]
            pred_label = 1 if any('maleza' in n for n in nombres_detectados) else 0
        except Exception:
            continue

        y_true.append(true_label)
        y_pred.append(pred_label)

    if not y_true:
        return jsonify({'error': 'No se pudieron evaluar imágenes válidas'}), 400

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return jsonify({
        'precision': precision,
        'recall': recall,
        'f1': f1
    })


if __name__ == '__main__':
    app.run(debug=True)
