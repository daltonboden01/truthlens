from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import os

app = Flask(__name__)
CORS(app)

print('Loading tokenizer from distilbert-base-uncased...')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

print('Loading model weights from daltonboden/truthlens-distilbert...')
model = DistilBertForSequenceClassification.from_pretrained('daltonboden/truthlens-distilbert')
model.eval()
print('Model loaded successfully.')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()
        confidence = round(probs[pred].item() * 100, 1)

    label = 'REAL' if pred == 1 else 'FAKE'
    verdict = label if confidence >= 85 else 'UNCERTAIN'

    return jsonify({
        'verdict': verdict,
        'label': label,
        'confidence': confidence,
        'fake_probability': round(probs[0].item() * 100, 1),
        'real_probability': round(probs[1].item() * 100, 1),
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'daltonboden/truthlens-distilbert'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
