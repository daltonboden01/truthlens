from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import os

app = Flask(__name__)
CORS(app)

# Load model and tokenizer from the /model folder
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model')

print('Loading model...')
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print('Model loaded successfully.')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512
    )

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()
        confidence = round(probs[pred].item() * 100, 1)

    label = 'REAL' if pred == 1 else 'FAKE'

    # Determine verdict category
    if confidence >= 85:
        verdict = label
    else:
        verdict = 'UNCERTAIN'

    return jsonify({
        'verdict': verdict,
        'label': label,
        'confidence': confidence,
        'fake_probability': round(probs[0].item() * 100, 1),
        'real_probability': round(probs[1].item() * 100, 1),
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'DistilBERT fine-tuned on WELFake'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
