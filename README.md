# ScopeSearch 🎬🔍
**Recuperación Semántica de Assets mediante CLIP + FAISS + LoRA/PEFT**

Motor de búsqueda multimodal que permite buscar imágenes/assets de VFX usando **lenguaje natural**:

> *"a dog running on the beach"* → devuelve los assets más similares en milisegundos.

---

## 🏗️ Arquitectura

```
CLIP (openai/clip-vit-base-patch32)
├── Vision Transformer (ViT-B/32)  ─── LoRA (r=8, α=16) → image embeddings
└── Text Encoder                   ─── LoRA (r=8, α=16) → text  embeddings
                                                   ↓
                               Contrastive Loss (Softmax CE, τ learnable)
                                                   ↓
                              FAISS IndexFlatIP (cosine similarity K-NN)
```

| Componente | Herramienta |
|---|---|
| Base Model | `openai/clip-vit-base-patch32` |
| Domain Adaptation | `peft` LoRA (~1% parámetros entrenables) |
| Vector DB | `faiss-cpu` (IndexFlatIP) |
| Métricas | Recall@K (1/5/10), MRR |
| UI | Streamlit |
| Package Manager | `uv` |

---

## 📁 Estructura

```
final_project 2/
├── data/
│   ├── raw/          ← assets originales (EXR, JPG, PNG...)
│   └── processed/    ← metadata CSV con pares image_path + description
├── scripts/
│   ├── download_coco_val.py     ← descarga COCO val 2017 (~1 GB)
│   ├── index_images.py          ← indexa imágenes → FAISS
│   └── search_cli.py            ← búsqueda por CLI (sin UI)
├── src/
│   ├── data/dataset.py          ← Dataset + DataLoader (CLIPProcessor)
│   ├── models/
│   │   ├── clip_lora.py         ← CLIP + LoRA via PEFT
│   │   └── loss.py              ← Contrastive Loss
│   ├── engine/
│   │   ├── train.py             ← Bucle de entrenamiento
│   │   └── evaluate.py          ← Recall@K y MRR
│   └── retrieval/faiss_index.py ← Índice FAISS
├── tests/
│   └── test_base_clip.py        ← Sanity check sin PEFT
├── vector_store/                ← Generado por index_images.py
│   ├── index.faiss
│   └── image_paths.json
├── app.py                       ← UI Streamlit
└── main.py                      ← CLI unificado (train/evaluate/search)
```

---

## ⚡ Quickstart

### 1. Entorno

```bash
uv venv
source .venv/bin/activate        # Linux/Mac (vast.ai)
# .venv\Scripts\activate          # Windows

uv pip install torch torchvision transformers peft faiss-cpu pillow pandas tqdm pytest streamlit
```

### 2. Sanity check (sin dataset, datos sintéticos)

```bash
uv run python -m pytest tests/test_base_clip.py -v
```

### 3. Descargar dataset COCO val 2017 (~1 GB, 5k imágenes)

```bash
uv run python scripts/download_coco_val.py
```
Genera automáticamente `data/processed/coco_val.csv`.

> **¿Tus propios assets VFX?** Saltá este paso y usá `--image_dir` en el siguiente.

### 4. Indexar imágenes (una sola vez, ~90 seg en GPU)

```bash
# Con CSV (recomendado)
uv run python scripts/index_images.py --metadata data/processed/coco_val.csv --batch_size 128

# Con tu propia carpeta de imágenes
uv run python scripts/index_images.py --image_dir data/raw/mis_assets/
```

### 5. Lanzar la UI

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Luego abrí `http://<IP>:8501` en el browser.

> **Túnel SSH (si el puerto no está expuesto):**
> ```bash
> ssh -p <PORT> -L 8501:localhost:8501 root@<IP>
> ```
> Luego `http://localhost:8501`.

### 6. Búsqueda por CLI (alternativa sin UI)

```bash
# One-shot
uv run python scripts/search_cli.py --query "a person cooking" --top_k 5

# Modo interactivo
uv run python scripts/search_cli.py
```

---

## 🎓 Fine-Tuning con LoRA (opcional)

Preparar un CSV con pares propios:
```csv
image_path,description
data/raw/sh010_specular.exr,"pase de specular de un vehículo metálico"
data/raw/smoke01.png,"humo denso con canal alfa premultiplicado"
```

Luego:
```bash
# Entrenar
uv run python main.py --mode train --metadata data/processed/vfx_dataset.csv --epochs 10

# Evaluar (Recall@K / MRR)
uv run python main.py --mode evaluate --metadata data/processed/vfx_dataset.csv
```

---

## 📐 Función de Pérdida

$$L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j)/\tau)}$$

La temperatura $\tau$ es un parámetro **aprendible** (`nn.Parameter`).

---

## 📊 Métricas de Evaluación

| Métrica | Descripción |
|---|---|
| **Recall@1** | % del asset correcto en el top-1 |
| **Recall@5** | % del asset correcto en el top-5 |
| **Recall@10** | % del asset correcto en el top-10 |
| **MRR** | Mean Reciprocal Rank |

### Baseline (CLIP sin fine-tuning, datos sintéticos)

```
test_forward_pass       PASSED  — image_embeds: (4, 512)
test_contrastive_loss   PASSED  — Loss = 1.2422
test_faiss_retrieval    PASSED  — Recall@1 = 50.00%
```

> Tras el fine-tuning con LoRA sobre datos reales de VFX, el Recall@1 debería mejorar significativamente.
