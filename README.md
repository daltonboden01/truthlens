# TruthLens — Misinformation Detector
**U-SOAR 2026 | Dalton Boden | IUP**

## Project Structure
```
truthlens/
├── app.py              # Flask API backend
├── index.html          # Frontend web app
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
├── README.md           # This file
└── model/              # Your trained DistilBERT model (you add this)
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── training_args.bin
    └── model.safetensors
```

## Setup

### Step 1 — Add your model files
Copy the files from your downloaded `distilbert_model_export.zip` into the `model/` folder.

### Step 2 — Run locally
```bash
pip install -r requirements.txt
python app.py
```
Then open `index.html` in your browser. It will connect to `http://localhost:5000`.

### Step 3 — Deploy to Render (free)
1. Push this folder to a GitHub repository
2. Go to render.com and sign up free
3. Click New → Web Service → connect your GitHub repo
4. Render will auto-detect render.yaml and deploy
5. Once deployed, copy your Render URL (e.g. https://truthlens-api.onrender.com)
6. Open index.html, find the line `const API_URL = 'http://localhost:5000'`
7. Replace it with your Render URL
8. Re-deploy or host index.html on GitHub Pages

## Model Info
- Architecture: DistilBERT fine-tuned for sequence classification
- Dataset: WELFake (72,134 news articles)
- F1 Score: 0.9931
- Training: 3 epochs, GPU T4, Kaggle Notebooks
