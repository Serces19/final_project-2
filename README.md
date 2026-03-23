# ScopeSearch 🎬🔍
**Recuperación Semántica de Assets en Pipelines de VFX mediante Adaptación de Dominio (PEFT)**

Motor de búsqueda multimodal que permite a artistas técnicos buscar assets de VFX (renders EXR, mattes, texturas, pases de Nuke) usando **lenguaje natural**, por ejemplo:

> *"pase de specular de un vehículo"* → devuelve los assets más similares en milisegundos.

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
| Package Manager | `uv` |

---

## 📁 Estructura

```
final_project 2/
├── data/
│   ├── raw/          ← assets originales (EXR, JPG, PNG...)
│   └── processed/    ← metadata CSV/JSON con pares image_path + description
├── src/
│   ├── data/dataset.py          ← Dataset + DataLoader (CLIPProcessor)
│   ├── models/
│   │   ├── clip_lora.py         ← CLIP + LoRA via PEFT
│   │   └── loss.py              ← Contrastive Loss personalizada
│   ├── engine/
│   │   ├── train.py             ← Bucle de entrenamiento
│   │   └── evaluate.py          ← Recall@K y MRR
│   └── retrieval/
│       └── faiss_index.py       ← Índice FAISS y búsqueda
├── tests/
│   └── test_base_clip.py        ← Sanity check sin PEFT
├── main.py                      ← CLI unificado
└── README.md
```

---

## ⚡ Quickstart

### 1. Entorno

```bash
uv venv
.venv\Scripts\activate
uv pip install torch torchvision transformers peft faiss-cpu pillow pandas tqdm pytest
```

### 2. Sanity Check (sin PEFT, datos sintéticos)

```bash
uv run python -m pytest tests/test_base_clip.py -v
```

Valida: forward pass ✅ | Contrastive Loss ✅ | FAISS Recall@1 ✅

### 3. Preparar Dataset

Crea un CSV con las columnas requeridas:

```csv
image_path,description
data/raw/sh010_specular.exr,"pase de specular de un vehículo metálico"
data/raw/smoke01.png,"humo denso con canal alfa premultiplicado"
data/raw/ckey_grn.jpg,"placa de chroma key verde con spill"
```

### 4. Entrenar con LoRA

```bash
uv run python main.py --mode train --metadata data/processed/vfx_dataset.csv --epochs 10
```

### 5. Evaluar (Recall@K / MRR)

```bash
uv run python main.py --mode evaluate --metadata data/processed/vfx_dataset.csv
```

### 6. Buscar un asset

```bash
uv run python main.py --mode search --query "pase de profundidad con niebla"
```

---

## 📐 Función de Pérdida

$$L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j)/\tau)}$$

La temperatura $\tau$ es un parámetro **aprendible** (`nn.Parameter`).

---

## 📊 Métricas de Evaluación

| Métrica | Descripción |
|---|---|
| **Recall@1** | % del asset correcto en el top-1 de resultados |
| **Recall@5** | % del asset correcto en el top-5 |
| **Recall@10** | % del asset correcto en el top-10 |
| **MRR** | Mean Reciprocal Rank — posición promedio del match correcto |

---

## 🧪 Hipótesis Central

> **Base CLIP** desconoce vocabulario técnico VFX (e.g., *"exr multipass"*, *"node graph"*).
> Tras el fine-tuning con **LoRA**, el Recall@1 y MRR deberían mejorar significativamente,
> demostrando la efectividad de la adaptación de dominio con PEFT.
