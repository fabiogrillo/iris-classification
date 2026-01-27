# Sentiment Analysis NLP - Master Plan

Documento completo per il progetto Sentiment Analysis con stack industry-grade. Da usare per weekly planning e tracking del progresso.

---

## Obiettivo del Progetto

Costruire un sistema end-to-end di classificazione sentiment (positive/negative) su movie reviews che dimostri:

1. **Competenze NLP**: da baseline classici a Transformers
2. **MLOps maturity**: experiment tracking, versioning, pipelines
3. **Production readiness**: API, containers, monitoring
4. **Industry tools**: stack usato in FAANG

**Dataset**: IMDB Movie Reviews (50,000 reviews, 25K train + 25K test, bilanciato)

---

## Perché Questo Progetto è Rilevante per FAANG

| Skill | Come lo dimostri | Dove si usa |
|-------|------------------|-------------|
| NLP/Transformers | Fine-tuning DistilBERT/RoBERTa | Google Search, Meta content moderation |
| MLOps | MLflow + DVC pipelines | Netflix ML Platform, Uber Michelangelo |
| Experiment Tracking | MLflow logging sistematico | Ogni team ML in FAANG |
| Data Labeling Workflow | Label Studio integration | Tesla Autopilot, Amazon Alexa |
| Model Serving | FastAPI + TorchServe | Real-time inference ovunque |
| Reproducibility | DVC + containerization | Requisito base in produzione |

---

# PARTE 1: TEORIA FONDAMENTALE

## 1.1 NLP Basics - Cosa Devi Sapere

### Cos'è il Natural Language Processing
NLP è l'intersezione tra linguistica e machine learning. L'obiettivo è far "capire" il linguaggio umano alle macchine.

**Sfide principali**:
- Il linguaggio è ambiguo ("bank" = banca o riva del fiume?)
- Il contesto cambia il significato ("not bad" = positivo!)
- Sarcasmo, ironia, slang

### Tokenization
**Cos'è**: Spezzare il testo in unità più piccole (tokens).

```
Input:  "I loved this movie!"
Output: ["I", "loved", "this", "movie", "!"]
```

**Tipi di tokenization**:
1. **Word-level**: ogni parola è un token
   - Pro: intuitivo
   - Contro: vocabulary enorme, parole rare = problemi

2. **Subword (BPE, WordPiece)**: divide parole in pezzi
   ```
   "unbelievable" → ["un", "##believ", "##able"]
   ```
   - Pro: vocabulary gestibile (~30K tokens), gestisce parole mai viste
   - Contro: meno intuitivo

3. **Character-level**: ogni carattere è un token
   - Pro: vocabulary minimo
   - Contro: sequenze lunghissime, perde significato

**Per i Transformers si usa Subword** (WordPiece per BERT, BPE per GPT).

### Embeddings
**Cos'è**: Rappresentazione numerica densa delle parole.

**Problema**: i computer capiscono solo numeri, non parole.

**Soluzione naive - One-Hot Encoding**:
```
vocabulary = ["cat", "dog", "movie"]
"cat"   → [1, 0, 0]
"dog"   → [0, 1, 0]
"movie" → [0, 0, 1]
```
Problemi:
- Vettori enormi (vocab 30K = vettori 30K dimensioni)
- Nessuna relazione semantica (cat e dog sono ugualmente distanti)

**Soluzione moderna - Word Embeddings**:
```
"cat"   → [0.2, -0.4, 0.7, ...]  (300 dimensioni)
"dog"   → [0.25, -0.35, 0.65, ...]  (simile a cat!)
"movie" → [-0.8, 0.1, 0.3, ...]  (diverso)
```

**Word2Vec** (2013):
- Addestra embeddings su grandi corpora
- "You shall know a word by the company it keeps"
- Famoso esempio: king - man + woman ≈ queen

**Problema di Word2Vec**: un embedding per parola, ma "bank" ha significati diversi!

---

## 1.2 Transformers - Il Cuore del Progetto

### Perché i Transformers hanno rivoluzionato NLP

**Prima (RNN/LSTM)**:
- Processano sequenze un token alla volta (lento)
- Difficoltà con dipendenze a lunga distanza
- "The movie that I watched yesterday with my friends was amazing"
  → LSTM fatica a collegare "movie" con "amazing"

**Dopo (Transformers, 2017)**:
- Processano tutti i token in parallelo (veloce, GPU-friendly)
- Ogni token può "guardare" direttamente ogni altro token
- Self-Attention: meccanismo che decide cosa è importante

### Self-Attention Spiegato

Immagina di leggere: "The animal didn't cross the street because it was too tired"

A cosa si riferisce "it"? All'animale o alla strada?

**Self-Attention calcola**:
1. Per ogni parola, crea 3 vettori: Query (Q), Key (K), Value (V)
2. Query di "it" chiede: "a chi mi riferisco?"
3. Keys di tutte le parole rispondono: "quanto sono rilevante?"
4. Attention scores = softmax(Q · K^T)
5. Output = somma pesata dei Values

Risultato: "it" avrà alta attention su "animal", bassa su "street".

```
           The  animal  didn't  cross  street  because  it  was  tired
it →       0.1   0.6     0.05   0.05   0.05    0.05   0.0  0.05  0.05
              ↑
          Alta attenzione!
```

### Architettura BERT

**BERT = Bidirectional Encoder Representations from Transformers**

```
[CLS] I loved this movie [SEP]
  ↓      ↓     ↓    ↓      ↓
┌─────────────────────────────┐
│      Transformer Encoder    │ ← 12 layers (BERT-base)
│      (Self-Attention × 12)  │ ← 768 hidden dimensions
└─────────────────────────────┘
  ↓      ↓     ↓    ↓      ↓
 H_CLS  H_I  H_loved ...   H_SEP
  ↓
Classification Head → Positive/Negative
```

**Token speciali**:
- `[CLS]`: token di classificazione, il suo output rappresenta l'intera sequenza
- `[SEP]`: separatore tra frasi

**Pre-training** (già fatto da Google/HuggingFace):
1. **Masked Language Model**: nascondi 15% delle parole, predici quali sono
2. **Next Sentence Prediction**: date 2 frasi, sono consecutive?

Risultato: BERT "capisce" il linguaggio dopo aver visto miliardi di parole.

**Fine-tuning** (quello che farai tu):
- Prendi BERT pre-trained
- Aggiungi classification head
- Allena su IMDB per 2-4 epoche
- BERT adatta la sua comprensione al sentiment

### DistilBERT vs BERT vs RoBERTa

| Modello | Parametri | Velocità | Performance | Quando usarlo |
|---------|-----------|----------|-------------|---------------|
| BERT-base | 110M | 1x | 100% | Baseline, risorse ok |
| DistilBERT | 66M | 1.6x | 97% | **Produzione, risorse limitate** |
| RoBERTa | 125M | 0.9x | 102% | Massima accuracy |

**Per il progetto**: inizia con DistilBERT (più veloce da trainare), poi prova RoBERTa.

---

## 1.3 MLOps - Perché è Fondamentale

### Il Problema che MLOps Risolve

**Scenario senza MLOps**:
```
"Ieri il modello aveva accuracy 92%, oggi 89%. Cosa è cambiato?"
"Non so, forse ho modificato qualcosa..."
"Quale versione del modello è in produzione?"
"Ehm... model_final_v2_FINAL_really_final.pkl?"
```

**Scenario con MLOps**:
```
"Run #47 ha accuracy 92% con lr=2e-5, epochs=3, batch=16"
"Run #48 ha accuracy 89% - ah, ho aumentato lr a 5e-5"
"Produzione usa model v1.2.0 dal registry, hash abc123"
```

### MLflow - Experiment Tracking

**Cos'è**: Piattaforma open-source per gestire il lifecycle ML.

**4 componenti**:
1. **Tracking**: logga parametri, metriche, artifacts
2. **Projects**: packaging riproducibile del codice
3. **Models**: formato standard per modelli
4. **Registry**: versioning e staging dei modelli

**Come si usa**:
```python
import mlflow

# Inizia un esperimento
mlflow.set_experiment("sentiment-analysis")

with mlflow.start_run(run_name="distilbert-v1"):
    # Log parametri
    mlflow.log_param("model", "distilbert-base-uncased")
    mlflow.log_param("learning_rate", 2e-5)
    mlflow.log_param("epochs", 3)
    mlflow.log_param("batch_size", 16)

    # Training...

    # Log metriche
    mlflow.log_metric("train_loss", 0.23)
    mlflow.log_metric("val_accuracy", 0.92)
    mlflow.log_metric("val_f1", 0.91)

    # Log modello
    mlflow.pytorch.log_model(model, "model")

    # Log artifacts (grafici, confusion matrix, etc)
    mlflow.log_artifact("confusion_matrix.png")
```

**UI**: MLflow fornisce una web UI per visualizzare tutti gli esperimenti.

### DVC - Data Version Control

**Cos'è**: Git per dati e modelli.

**Problema**: Git non gestisce bene file grandi (dataset, modelli).

**Soluzione DVC**:
```bash
# Invece di committare il file (troppo grande)
git add data/imdb.csv  # ❌ file da 100MB

# DVC traccia il file e lo mette in storage esterno
dvc add data/imdb.csv  # ✅ crea imdb.csv.dvc (pochi KB)
git add data/imdb.csv.dvc  # committa solo il puntatore
```

**DVC Pipelines** - definisci il workflow ML:
```yaml
# dvc.yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - data/raw/imdb.csv
      - src/preprocess.py
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  train:
    cmd: python src/train.py
    deps:
      - data/processed/
      - src/train.py
      - configs/train.yaml
    outs:
      - models/distilbert/
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - models/distilbert/
      - data/processed/test.csv
    metrics:
      - evaluation.json:
          cache: false
```

**Eseguire la pipeline**:
```bash
dvc repro  # esegue solo gli stage modificati
```

**Beneficio**: riproducibilità totale. `git checkout v1.0` + `dvc checkout` = esatto stato del progetto.

### Label Studio - Data Labeling

**Cos'è**: Piattaforma open-source per annotare dati.

**Perché è importante**:
- In produzione, i dati spesso non sono già labellati
- Serve un workflow strutturato per annotatori
- Quality control delle annotazioni

**Per questo progetto**:
- IMDB è già labellato, MA
- Dimostri di conoscere il workflow professionale
- Crei demo con subset di dati

**Setup**:
```bash
docker run -p 8080:8080 heartexlabs/label-studio
```

**Interfaccia per sentiment**:
```xml
<View>
  <Text name="text" value="$review"/>
  <Choices name="sentiment" toName="text" choice="single">
    <Choice value="positive"/>
    <Choice value="negative"/>
  </Choices>
</View>
```

---

## 1.4 Model Serving - Da Notebook a Production

### FastAPI per ML

**Perché FastAPI**:
- Async nativo (gestisce molte richieste)
- Validazione automatica (Pydantic)
- Documentazione automatica (Swagger)
- Type hints = meno bug

**Struttura base**:
```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
classifier = pipeline("sentiment-analysis", model="./models/distilbert")

class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ReviewRequest):
    result = classifier(request.text)[0]
    return PredictionResponse(
        sentiment=result["label"],
        confidence=result["score"]
    )
```

### TorchServe per Batch Inference

**Cos'è**: Framework di serving di PyTorch, ottimizzato per produzione.

**Quando usarlo**:
- Alto throughput richiesto
- Batching automatico
- Model versioning nativo

**Per questo progetto**: FastAPI per demo, menzione TorchServe per scalabilità.

### Docker Compose per Multi-Service

```yaml
version: '3.8'

services:
  api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/distilbert
    volumes:
      - ./models:/models

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.0
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0

  label-studio:
    image: heartexlabs/label-studio:latest
    ports:
      - "8080:8080"
    volumes:
      - ./label-studio-data:/label-studio/data
```

```bash
docker-compose up -d  # avvia tutto
```

---

# PARTE 2: PIANO DI IMPLEMENTAZIONE

## Struttura Temporale Suggerita

| Settimana | Focus | Output |
|-----------|-------|--------|
| 1 | Setup + EDA | Ambiente pronto, notebook EDA |
| 2 | Baselines + MLflow | TF-IDF e LSTM loggati in MLflow |
| 3 | Transformers | DistilBERT fine-tuned, confronto modelli |
| 4 | MLOps Pipeline | DVC pipeline completa |
| 5 | Deployment + Docs | API, Docker, README finale |

---

## Settimana 1: Setup & Exploratory Data Analysis

### Obiettivi
- [x] Creare struttura progetto ✅ 2026-01-21
- [x] Setup ambiente (venv, requirements) ✅ 2026-01-21
- [x] Inizializzare DVC ✅ 2026-01-21
- [x] Download e esplorazione dataset IMDB ✅ 2026-01-21
- [x] Notebook EDA completo ✅ 2026-01-21
- [x] Setup Label Studio (demo) ✅ 2026-01-21

### Step-by-Step

#### 1.1 Creare Struttura Progetto

```bash
mkdir -p Sentiment-Analysis-NLP/{data/{raw,processed,labeled},notebooks,src/{data,models,training,inference},configs,tests}
cd Sentiment-Analysis-NLP
```

Struttura finale:
```
Sentiment-Analysis-NLP/
├── data/
│   ├── raw/                 # Dataset originale
│   ├── processed/           # Dopo preprocessing
│   └── labeled/             # Export Label Studio
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_transformers.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py
│   └── inference/
│       ├── __init__.py
│       └── predict.py
├── configs/
│   └── train.yaml
├── tests/
├── mlruns/                  # MLflow artifacts
├── app.py                   # FastAPI
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

#### 1.2 Setup Ambiente

```bash
python -m venv venv
source venv/bin/activate

# requirements.txt
pip install torch transformers datasets
pip install scikit-learn pandas numpy matplotlib seaborn
pip install mlflow dvc
pip install fastapi uvicorn pydantic
pip install jupyter ipykernel
pip install pytest

pip freeze > requirements.txt
```

#### 1.3 Inizializzare DVC

```bash
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"

# Opzionale: remote storage (S3, GCS, o local)
dvc remote add -d myremote /path/to/storage
```

#### 1.4 Download Dataset IMDB

```python
# src/data/download.py
from datasets import load_dataset
import pandas as pd

def download_imdb():
    """Download IMDB dataset from Hugging Face."""
    dataset = load_dataset("imdb")

    # Convert to pandas
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    # Save
    train_df.to_csv("data/raw/train.csv", index=False)
    test_df.to_csv("data/raw/test.csv", index=False)

    print(f"Train: {len(train_df)} samples")
    print(f"Test: {len(test_df)} samples")

if __name__ == "__main__":
    download_imdb()
```

```bash
python src/data/download.py
dvc add data/raw/train.csv data/raw/test.csv
git add data/raw/*.dvc data/raw/.gitignore
```

#### 1.5 Notebook EDA

```python
# notebooks/01_eda.ipynb

# Cell 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Cell 2: Load data
train_df = pd.read_csv("../data/raw/train.csv")
test_df = pd.read_csv("../data/raw/test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(train_df.head())

# Cell 3: Class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

train_df['label'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Train Set Distribution')
axes[0].set_xticklabels(['Negative', 'Positive'], rotation=0)

test_df['label'].value_counts().plot(kind='bar', ax=axes[1], color=['green', 'red'])
axes[1].set_title('Test Set Distribution')
axes[1].set_xticklabels(['Negative', 'Positive'], rotation=0)

plt.tight_layout()
plt.savefig('../figures/class_distribution.png')

# Cell 4: Review length analysis
train_df['word_count'] = train_df['text'].apply(lambda x: len(x.split()))
train_df['char_count'] = train_df['text'].apply(len)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(train_df['word_count'], bins=50, edgecolor='black')
axes[0].set_xlabel('Word Count')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Review Length (Words)')
axes[0].axvline(train_df['word_count'].mean(), color='red', linestyle='--', label=f'Mean: {train_df["word_count"].mean():.0f}')
axes[0].legend()

# By sentiment
train_df[train_df['label']==1]['word_count'].hist(ax=axes[1], bins=50, alpha=0.5, label='Positive', color='green')
train_df[train_df['label']==0]['word_count'].hist(ax=axes[1], bins=50, alpha=0.5, label='Negative', color='red')
axes[1].set_xlabel('Word Count')
axes[1].set_title('Review Length by Sentiment')
axes[1].legend()

plt.tight_layout()
plt.savefig('../figures/length_distribution.png')

# Cell 5: Most common words
from wordcloud import WordCloud

positive_text = ' '.join(train_df[train_df['label']==1]['text'].tolist())
negative_text = ' '.join(train_df[train_df['label']==0]['text'].tolist())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

wc_pos = WordCloud(width=600, height=400, background_color='white').generate(positive_text)
axes[0].imshow(wc_pos, interpolation='bilinear')
axes[0].set_title('Positive Reviews - Word Cloud')
axes[0].axis('off')

wc_neg = WordCloud(width=600, height=400, background_color='white').generate(negative_text)
axes[1].imshow(wc_neg, interpolation='bilinear')
axes[1].set_title('Negative Reviews - Word Cloud')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('../figures/wordclouds.png')

# Cell 6: Sample reviews
print("=== POSITIVE EXAMPLES ===")
for i, row in train_df[train_df['label']==1].head(3).iterrows():
    print(f"\n{row['text'][:300]}...")

print("\n=== NEGATIVE EXAMPLES ===")
for i, row in train_df[train_df['label']==0].head(3).iterrows():
    print(f"\n{row['text'][:300]}...")

# Cell 7: Statistics summary
stats = {
    'Total Samples': len(train_df),
    'Positive Samples': (train_df['label']==1).sum(),
    'Negative Samples': (train_df['label']==0).sum(),
    'Avg Words per Review': train_df['word_count'].mean(),
    'Max Words': train_df['word_count'].max(),
    'Min Words': train_df['word_count'].min(),
}
pd.DataFrame([stats]).T.rename(columns={0: 'Value'})
```

#### 1.6 Setup Label Studio (Demo)

```bash
docker run -d -p 8080:8080 \
  -v $(pwd)/label-studio-data:/label-studio/data \
  --name label-studio \
  heartexlabs/label-studio:latest
```

Poi:
1. Vai a `http://localhost:8080`
2. Crea account
3. Crea progetto "Sentiment Demo"
4. Importa 100 samples da train.csv
5. Configura labeling interface per sentiment
6. Annota 20-30 samples come demo
7. Screenshot per README

---

## Settimana 2: Baseline Models + MLflow

### Obiettivi
- [ ] Setup MLflow tracking
- [ ] Implementare TF-IDF + Logistic Regression
- [ ] Implementare Word2Vec + LSTM
- [ ] Loggare tutti gli esperimenti in MLflow
- [ ] Confrontare risultati

### Step-by-Step

#### 2.1 Configurare MLflow

```python
# src/training/mlflow_config.py
import mlflow
import os

def setup_mlflow(experiment_name: str = "sentiment-analysis"):
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

    # Abilita autolog per sklearn e pytorch
    mlflow.sklearn.autolog()

    return mlflow

# Avviare UI
# mlflow ui --port 5000
```

#### 2.2 Preprocessing per Baselines

```python
# src/data/preprocess.py
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters (keep letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def preprocess_dataset(input_path: str, output_path: str):
    """Preprocess and save dataset."""
    df = pd.read_csv(input_path)

    # Clean text
    df['text_clean'] = df['text'].apply(clean_text)

    # Remove empty
    df = df[df['text_clean'].str.len() > 0]

    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")

if __name__ == "__main__":
    preprocess_dataset("data/raw/train.csv", "data/processed/train.csv")
    preprocess_dataset("data/raw/test.csv", "data/processed/test.csv")
```

#### 2.3 Baseline 1: TF-IDF + Logistic Regression

```python
# notebooks/02_baselines.ipynb

# Cell 1: Setup
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("file:../mlruns")
mlflow.set_experiment("sentiment-baselines")

# Cell 2: Load data
train_df = pd.read_csv("../data/processed/train.csv")
test_df = pd.read_csv("../data/processed/test.csv")

X_train = train_df['text_clean']
y_train = train_df['label']
X_test = test_df['text_clean']
y_test = test_df['label']

# Cell 3: TF-IDF + Logistic Regression
with mlflow.start_run(run_name="tfidf-logreg"):
    # Parameters
    max_features = 10000
    ngram_range = (1, 2)
    C = 1.0

    mlflow.log_param("vectorizer", "tfidf")
    mlflow.log_param("max_features", max_features)
    mlflow.log_param("ngram_range", str(ngram_range))
    mlflow.log_param("model", "logistic_regression")
    mlflow.log_param("C", C)

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'TF-IDF + LogReg - Accuracy: {accuracy:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('../figures/cm_tfidf_logreg.png')
    mlflow.log_artifact('../figures/cm_tfidf_logreg.png')

    # Log model
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))
```

#### 2.4 Baseline 2: Word2Vec + LSTM

```python
# Cell 4: LSTM with Word2Vec embeddings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
import numpy as np

# Tokenize
def tokenize(text):
    return text.split()

train_tokens = [tokenize(text) for text in X_train]
test_tokens = [tokenize(text) for text in X_test]

# Train Word2Vec
w2v_model = Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=2, workers=4)

# Convert to sequences
def text_to_sequence(tokens, w2v, max_len=200):
    seq = []
    for token in tokens[:max_len]:
        if token in w2v.wv:
            seq.append(w2v.wv[token])
        else:
            seq.append(np.zeros(100))

    # Pad
    while len(seq) < max_len:
        seq.append(np.zeros(100))

    return np.array(seq)

X_train_seq = np.array([text_to_sequence(t, w2v_model) for t in train_tokens])
X_test_seq = np.array([text_to_sequence(t, w2v_model) for t in test_tokens])

# LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        return out

# Training loop (simplified)
with mlflow.start_run(run_name="word2vec-lstm"):
    mlflow.log_param("embedding", "word2vec")
    mlflow.log_param("embedding_dim", 100)
    mlflow.log_param("model", "lstm")
    mlflow.log_param("hidden_dim", 128)
    mlflow.log_param("n_layers", 2)
    mlflow.log_param("epochs", 5)

    # ... training code ...

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.pytorch.log_model(model, "model")
```

---

## Settimana 3: Transformers

### Obiettivi
- [x] Fine-tune DistilBERT ✅ 92.7% accuracy
- [ ] Model Evaluation (confusion matrix, classification report)
- [ ] Inference su review custom
- [ ] Confronto con baselines in MLflow

### Step-by-Step

#### 3.1 DistilBERT Fine-tuning

```python
# notebooks/03_transformers.ipynb

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import evaluate
import mlflow

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Prepare dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/distilbert",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=predictions, references=labels)["f1"]
    }

# Train with MLflow
with mlflow.start_run(run_name="distilbert-finetuned"):
    mlflow.log_param("model", model_name)
    mlflow.log_param("epochs", training_args.num_train_epochs)
    mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
    mlflow.log_param("learning_rate", training_args.learning_rate)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate
    results = trainer.evaluate()
    mlflow.log_metrics(results)

    # Save model
    trainer.save_model("./models/distilbert-final")
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="model"
    )
```

#### 3.2 Model Evaluation

**Perché è importante**: La sola accuracy non basta. Confusion matrix e classification report mostrano dove il modello sbaglia, utile per debugging e per spiegare i risultati in un'intervista.

**Cosa faremo**:
1. Generare predizioni sul test set
2. Creare confusion matrix per visualizzare errori (False Positives vs False Negatives)
3. Calcolare precision, recall, F1 per ogni classe

```python
# notebooks/03_transformers.ipynb - Nuova cella dopo il training

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Genera predizioni sul test set
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(axis=-1)
y_true = test_df['label'].values

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'DistilBERT Confusion Matrix - Accuracy: {(y_pred == y_true).mean():.2%}')
plt.savefig('../figures/confusion_matrix_distilbert.png')
plt.show()

# Classification Report
print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

# Log artifact in MLflow (opzionale)
mlflow.log_artifact('../figures/confusion_matrix_distilbert.png')
```

**Output atteso**:
- Confusion matrix con ~92-93% accuracy
- Precision/Recall bilanciati (dataset IMDB è bilanciato)
- Pochi errori rispetto ai baseline

#### 3.3 Inference su Review Custom

**Perché è importante**: Dimostra che il modello funziona su dati reali, non solo sul test set. Essenziale per demo e per capire il comportamento del modello su edge cases.

**Cosa faremo**:
1. Caricare il modello salvato usando `pipeline` di Hugging Face
2. Testare su review inventate (positive, negative, ambigue)
3. Analizzare i risultati e la confidence

```python
# notebooks/03_transformers.ipynb - Nuova cella per inference

from transformers import pipeline

# Carica il modello salvato (non serve rifare training!)
classifier = pipeline(
    "sentiment-analysis",
    model="../models/distilbert-final",
    tokenizer="distilbert-base-uncased"
)

# Test su review custom
test_reviews = [
    "This movie was absolutely fantastic! Best film I've seen this year.",
    "Terrible acting, boring plot. Complete waste of time.",
    "It was okay, nothing special but not bad either.",
    "The cinematography was beautiful but the story made no sense.",
    "I fell asleep halfway through. So boring!",
    "A masterpiece! The director outdid himself.",
]

# Predizioni
print("=" * 60)
print("INFERENCE RESULTS")
print("=" * 60)
for review in test_reviews:
    result = classifier(review)[0]
    label = result['label']
    score = result['score']
    sentiment = "POSITIVE" if label == "LABEL_1" else "NEGATIVE"
    print(f"\nReview: {review[:50]}...")
    print(f"Prediction: {sentiment} (confidence: {score:.2%})")
```

**Output atteso**:
```
Review: This movie was absolutely fantastic! Best film...
Prediction: POSITIVE (confidence: 99.2%)

Review: Terrible acting, boring plot. Complete waste...
Prediction: NEGATIVE (confidence: 98.7%)

Review: It was okay, nothing special but not bad either...
Prediction: POSITIVE (confidence: 62.3%)  # Caso ambiguo - confidence bassa
```

**Nota**: La confidence bassa sui casi ambigui è un buon segno - il modello "sa di non sapere".

#### 3.4 Confronto Finale con Baselines

**Perché è importante**: Documenta il miglioramento rispetto ai baseline e giustifica la scelta del modello finale.

```python
# Tabella riassuntiva
import pandas as pd

results_df = pd.DataFrame({
    'Model': ['TF-IDF + LogReg', 'Word2Vec + BiLSTM', 'DistilBERT'],
    'Accuracy': [0.892, 0.851, 0.927],
    'F1 Score': [0.89, 0.85, 0.93],
    'Training Time': ['~30 sec', '~5 min', '~20 min'],
    'Inference Speed': ['Fast', 'Medium', 'Slow'],
    'GPU Required': ['No', 'Yes', 'Yes']
})

print(results_df.to_markdown(index=False))
```

**Conclusioni da documentare**:
1. **DistilBERT vince** con +3.5% accuracy rispetto al miglior baseline
2. **Trade-off**: più lento ma più accurato
3. **Per produzione**: DistilBERT è il giusto compromesso velocità/accuracy

---

## Settimana 4: MLOps Pipeline

### Obiettivi
- [ ] Creare DVC pipeline completa
- [ ] Setup MLflow Model Registry
- [ ] Creare config files YAML
- [ ] Test pipeline reproducibility

### Step-by-Step

#### 4.1 DVC Pipeline

```yaml
# dvc.yaml
stages:
  download:
    cmd: python src/data/download.py
    outs:
      - data/raw/train.csv
      - data/raw/test.csv

  preprocess:
    cmd: python src/data/preprocess.py
    deps:
      - data/raw/train.csv
      - data/raw/test.csv
      - src/data/preprocess.py
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  train_baseline:
    cmd: python src/training/train_baseline.py
    deps:
      - data/processed/train.csv
      - src/training/train_baseline.py
      - configs/baseline.yaml
    outs:
      - models/tfidf_logreg/
    metrics:
      - metrics/baseline.json:
          cache: false

  train_transformer:
    cmd: python src/training/train_transformer.py
    deps:
      - data/processed/train.csv
      - src/training/train_transformer.py
      - configs/transformer.yaml
    outs:
      - models/distilbert/
    metrics:
      - metrics/transformer.json:
          cache: false

  evaluate:
    cmd: python src/training/evaluate.py
    deps:
      - models/distilbert/
      - data/processed/test.csv
    metrics:
      - metrics/evaluation.json:
          cache: false
    plots:
      - plots/confusion_matrix.png
      - plots/roc_curve.png
```

```bash
# Eseguire pipeline
dvc repro

# Visualizzare DAG
dvc dag

# Confrontare metriche tra versioni
dvc metrics diff
```

#### 4.2 MLflow Model Registry

```python
# src/training/register_model.py
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Registrare modello dal run migliore
run_id = "abc123..."  # ID del run migliore
model_uri = f"runs:/{run_id}/model"

# Registra nel registry
mlflow.register_model(model_uri, "sentiment-classifier")

# Transizione a Production
client.transition_model_version_stage(
    name="sentiment-classifier",
    version=1,
    stage="Production"
)

# Caricare modello in produzione
model = mlflow.pyfunc.load_model("models:/sentiment-classifier/Production")
```

---

## Settimana 5: Deployment & Documentation

### Obiettivi
- [ ] FastAPI application completa
- [ ] Docker + docker-compose
- [ ] Load testing con Locust
- [ ] README professionale
- [ ] Model explainability (SHAP)

### Step-by-Step

#### 5.1 FastAPI Application

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import mlflow
from transformers import pipeline

app = FastAPI(
    title="Sentiment Analysis API",
    description="Production-ready sentiment classification using DistilBERT",
    version="1.0.0"
)

# Load model from MLflow registry
model_uri = "models:/sentiment-classifier/Production"
classifier = mlflow.pyfunc.load_model(model_uri)

# Alternative: load directly from Hugging Face
# classifier = pipeline("sentiment-analysis", model="./models/distilbert-final")

class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, example="This movie was amazing!")

class BatchReviewRequest(BaseModel):
    reviews: List[str] = Field(..., min_items=1, max_items=32)

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "distilbert-sentiment"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ReviewRequest):
    try:
        result = classifier.predict(request.text)
        return PredictionResponse(
            sentiment="positive" if result[0] == 1 else "negative",
            confidence=float(result[1])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchReviewRequest):
    predictions = []
    for text in request.reviews:
        result = classifier.predict(text)
        predictions.append(PredictionResponse(
            sentiment="positive" if result[0] == 1 else "negative",
            confidence=float(result[1])
        ))
    return BatchPredictionResponse(predictions=predictions)
```

#### 5.2 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.0
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow/mlruns
    command: >
      mlflow server
      --backend-store-uri sqlite:///mlflow.db
      --default-artifact-root /mlflow/mlruns
      --host 0.0.0.0

  label-studio:
    image: heartexlabs/label-studio:latest
    ports:
      - "8080:8080"
    volumes:
      - ./label-studio-data:/label-studio/data
```

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 5.3 Model Explainability (SHAP)

```python
# src/inference/explain.py
import shap
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="./models/distilbert-final")

# Create explainer
explainer = shap.Explainer(classifier)

# Explain predictions
texts = [
    "This movie was absolutely fantastic!",
    "Terrible acting and boring plot.",
    "Not bad, but could have been better."
]

shap_values = explainer(texts)

# Visualize
shap.plots.text(shap_values[0])  # Per ogni testo
shap.plots.bar(shap_values)      # Summary
```

---

# PARTE 3: CHICCHE FAANG - COME DISTINGUERSI

## Checklist Elementi Distintivi

### Must Have (Essenziali)
- [ ] MLflow experiment tracking con confronto modelli
- [ ] DVC pipeline per reproducibility
- [ ] Hugging Face Transformers (DistilBERT)
- [ ] FastAPI production API
- [ ] Docker containerization
- [ ] README professionale con metriche

### Nice to Have (Distintivi)
- [ ] Label Studio integration (workflow annotation)
- [ ] Hyperparameter tuning con Optuna
- [ ] Model Registry con versioning
- [ ] SHAP explainability
- [ ] Load testing con Locust
- [ ] CI/CD con GitHub Actions

### Wow Factor (Eccellenza)
- [ ] Push modello su Hugging Face Hub (pubblico)
- [ ] Evidently AI per drift detection
- [ ] A/B testing simulation
- [ ] Prometheus + Grafana monitoring
- [ ] Kubernetes deployment (Helm chart)

---

## Come Raccontarlo nel CV/Interview

### Resume Bullet Points
```
• Developed production NLP sentiment classifier achieving 93% accuracy using
  DistilBERT fine-tuning, with MLflow tracking across 50+ experiments

• Implemented end-to-end MLOps pipeline using DVC for data versioning and
  reproducible training, reducing experiment setup time by 80%

• Deployed containerized inference API serving 100+ req/s with p99 latency < 200ms

• Integrated Label Studio for annotation workflow, demonstrating understanding
  of real-world data labeling challenges
```

### Interview Talking Points
1. **Trade-offs**: "Ho scelto DistilBERT invece di BERT-large perché..."
2. **Scale considerations**: "In produzione, userei TorchServe per batching..."
3. **Monitoring**: "Ho implementato drift detection perché in produzione..."
4. **Lessons learned**: "Il debugging più difficile è stato quando..."

---

# PARTE 4: RISORSE PER APPRENDIMENTO

## Priorità Alta (Prima di iniziare)

1. **Hugging Face Course** (gratuito, 4-6 ore)
   - https://huggingface.co/course
   - Capitoli 1-4 per basi Transformers
   - Capitolo 5 per datasets

2. **MLflow Quickstart** (2 ore)
   - https://mlflow.org/docs/latest/quickstart.html
   - Focus su tracking e model registry

3. **DVC Tutorial** (2 ore)
   - https://dvc.org/doc/start
   - Focus su versioning e pipelines

## Priorità Media (Durante il progetto)

4. **FastAPI Tutorial** (2 ore)
   - https://fastapi.tiangolo.com/tutorial/
   - Se non già familiare

5. **Docker Compose** (1 ora)
   - https://docs.docker.com/compose/gettingstarted/

## Priorità Bassa (Dopo completamento base)

6. **SHAP Documentation**
   - https://shap.readthedocs.io/

7. **Evidently AI Quickstart**
   - https://docs.evidentlyai.com/

---

# PARTE 5: TRACKING PROGRESS

## Checklist Settimanale

### Settimana 1
- [ ] Setup progetto completato
- [ ] DVC inizializzato
- [ ] Dataset scaricato e versionato
- [ ] Notebook EDA completato con visualizzazioni
- [ ] Label Studio funzionante (demo)
- [ ] Prima commit con struttura progetto

### Settimana 2
- [ ] MLflow configurato
- [ ] TF-IDF baseline implementato e loggato
- [ ] LSTM baseline implementato e loggato
- [ ] Confronto modelli in MLflow UI
- [ ] Metriche baseline documentate

### Settimana 3
- [x] DistilBERT fine-tuned (92.7% accuracy)
- [ ] Model Evaluation (confusion matrix, classification report)
- [ ] Inference su review custom
- [ ] Confronto finale con baselines documentato

### Settimana 4
- [ ] DVC pipeline completa (dvc.yaml)
- [ ] Pipeline testata (dvc repro)
- [ ] Model Registry configurato
- [ ] Modello registrato come "Production"

### Settimana 5
- [ ] FastAPI application completa
- [ ] Docker + docker-compose funzionante
- [ ] README professionale
- [ ] (Opzionale) SHAP explainability
- [ ] (Opzionale) Modello su Hugging Face Hub
- [ ] Progetto pushato su GitHub

---

## Note per Weekly Planning con Claude

Quando usi questo documento per il weekly planning:

1. **Indica la settimana corrente**: "Sono alla settimana 2 del progetto Sentiment Analysis"

2. **Riporta il progresso**: "Ho completato X, Y, Z. Mi manca W"

3. **Chiedi supporto specifico**: "Ho difficoltà con il setup di MLflow, puoi aiutarmi?"

4. **Review del codice**: "Ecco il mio train.py, è strutturato bene per MLflow?"

5. **Troubleshooting**: "Ho questo errore durante il fine-tuning di DistilBERT..."

Il documento è strutturato per essere un riferimento completo. Non serve leggerlo tutto in una sessione - usalo come guida progressiva settimana per settimana.
