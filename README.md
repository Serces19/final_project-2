# ScopeSearch
**Recuperación Semántica de Assets mediante CLIP + FAISS + LoRA/PEFT**

Motor de búsqueda multimodal para buscar imágenes por **texto** o por **imagen similar**, sin depender de nombres de archivo ni metadatos manuales.

---

## Arquitectura

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
| Métricas | Recall@1, training loss por epoch |
| UI | Streamlit (`app.py`) |
| Package Manager | `uv` |

---

## Estructura del Proyecto

```
final_project 2/
├── data/
│   ├── raw/
│   │   ├── coco/val2017/          ← COCO val 2017 (5k imágenes)
│   │   └── vfx_assets/            ← Assets VFX propios
│   │       ├── assets/
│   │       ├── chroma/
│   │       ├── depth/
│   │       └── normals/
│   └── processed/
│       ├── coco_val.csv           ← Generado por download_coco_val.py
│       ├── vfx_dataset.csv        ← Generado por build_csv_from_folders.py
│       ├── vfx_train.csv          ← Split de entrenamiento
│       ├── vfx_val.csv            ← Split de validación
│       └── combined.csv           ← COCO + VFX unificados
├── scripts/
│   ├── download_coco_val.py       ← Descarga COCO val 2017
│   ├── build_csv_from_folders.py  ← Genera CSV desde carpetas VFX propias
│   ├── validate_dataset.py        ← Valida el CSV antes de entrenar
│   ├── index_images.py            ← Crea el índice FAISS
│   ├── search_cli.py              ← Búsqueda por CLI (texto o imagen)
│   ├── prepare_vfx_dataset.py     ← Split train/val
│   └── plot_metrics.py            ← Genera gráficas de entrenamiento
├── src/
│   ├── data/dataset.py
│   ├── models/clip_lora.py        ← CLIP + LoRA via PEFT
│   ├── models/loss.py             ← Contrastive Loss
│   ├── engine/train.py            ← Bucle de entrenamiento + métricas
│   └── engine/evaluate.py         ← Recall@K y MRR
├── checkpoints/                   ← Pesos LoRA guardados tras entrenar
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── vector_store/                  ← Generado por index_images.py
│   ├── index.faiss
│   ├── image_paths.json
│   └── meta.json                  ← Indica qué modelo generó el índice
├── logs/
│   ├── training_metrics.json      ← Historial época × loss × Recall@1
│   └── training_curves.png        ← Gráficas generadas por plot_metrics.py
├── tests/test_base_clip.py        ← Sanity check sin PEFT (baseline)
├── app.py                         ← UI Streamlit
└── main.py                        ← CLI unificado (train / evaluate)
```

---

## Quickstart — Entorno

```bash
git clone https://github.com/Serces19/final_project-2.git
cd final_project-2

uv venv
source .venv/bin/activate   # Linux/Mac

uv pip install torch torchvision transformers peft faiss-cpu \
               pillow pandas tqdm pytest streamlit matplotlib
```

---

## Paso 1 — Construir la Base de Conocimiento (Índice)

El índice FAISS puede contener **cualquier combinación de datasets**. Se puede usar el modelo base o el fine-tuned.

### Opción A — Solo COCO val 2017 (5k imágenes generales)

```bash
# Descargar
uv run python scripts/download_coco_val.py

# Indexar
uv run python scripts/index_images.py \
    --metadata data/processed/coco_val.csv \
    --batch_size 128
```

### Opción B — Solo assets VFX propios

```bash
# Generar CSV desde tus carpetas (assets/, chroma/, depth/, normals/)
uv run python scripts/build_csv_from_folders.py \
    --root data/raw/vfx_assets/ \
    --output data/processed/vfx_dataset.csv \
    --augment 3

# Indexar
uv run python scripts/index_images.py \
    --metadata data/processed/vfx_dataset.csv \
    --batch_size 128
```

### Opción C — COCO + VFX juntos en el mismo índice (RECOMENDADO)

```bash
# Combinar los dos CSVs en uno
uv run python -c "
import pandas as pd
coco = pd.read_csv('data/processed/coco_val.csv')
vfx  = pd.read_csv('data/processed/vfx_dataset.csv')
combined = pd.concat([coco, vfx], ignore_index=True).sample(frac=1, random_state=42)
combined.to_csv('data/processed/combined.csv', index=False)
print(f'Total: {len(combined)} pares ({len(coco)} COCO + {len(vfx)} VFX)')
"

# Indexar todo junto
uv run python scripts/index_images.py \
    --metadata data/processed/combined.csv \
    --batch_size 128
```

> Para usar el modelo fine-tuned (después de entrenar), agregá `--checkpoint checkpoints/`:
> ```bash
> uv run python scripts/index_images.py \
>     --metadata data/processed/combined.csv \
>     --checkpoint checkpoints/ \
>     --batch_size 128
> ```

---

## Paso 2 — Sanity Check (Baseline sin entrenamiento)

```bash
uv run python -m pytest tests/test_base_clip.py -v
```

Resultados esperados:
```
test_forward_pass       PASSED  — image_embeds: (4, 512)
test_contrastive_loss   PASSED  — Loss ≈ 1.24
test_faiss_retrieval    PASSED  — Recall@1 = 50.00%  ← baseline
```

---

## Paso 3 — Preparar Dataset VFX para Fine-Tuning

### 3a. Validar el CSV antes de entrenar

```bash

uv run python scripts/build_csv_from_folders.py \
    --root data/raw/vfx_assets/ \
    --output data/processed/vfx_dataset.csv \
    --augment 3

uv run python scripts/validate_dataset.py --csv data/processed/vfx_dataset.csv
```

Verifica: archivos existentes, imágenes legibles, longitud de captions, distribución por categoría.

### 3b. Split train / val (80% / 20%)

```bash
uv run python scripts/prepare_vfx_dataset.py \
    --split --input data/processed/vfx_dataset.csv
# → data/processed/vfx_train.csv
# → data/processed/vfx_val.csv
```

---

## Paso 4 — Entrenar con LoRA

```bash
uv run python main.py \
    --mode train \
    --metadata data/processed/vfx_train.csv \
    --val      data/processed/vfx_val.csv \
    --epochs   15 \
    --batch_size 32 \
    --lr       1e-4 \
    --lora_r   16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --weight_decay 0.05 \
    --grad_clip 1.0 \
    --label_smoothing 0.1 \
    --scheduler cosine \
    --warmup_epochs 2

```

Durante el entrenamiento verás una tabla en tiempo real:

```
Epoch | Train Loss | Recall@1 (val)
──────────────────────────────────
    1 |     1.8234 |         12.50%
    5 |     1.4120 |         37.50%
   10 |     1.1890 |         62.50%
══════════════════════════════════
  Best loss  : 1.1890 (epoch 10)
  Best R@1   : 62.50% (epoch 10)
══════════════════════════════════
Metrics saved: logs/training_metrics.json
```

Los pesos LoRA se guardan en `checkpoints/` (~4-8 MB).

### Generar gráficas de entrenamiento

```bash
uv pip install matplotlib
uv run python scripts/plot_metrics.py
# → logs/training_curves.png
```

---

## Paso 5 — Inferencia / Búsqueda

### UI Streamlit

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Abrí `http://<IP>:8501` en el browser.

> **Túnel SSH si el puerto no está expuesto:**
> ```bash
> ssh -p <PORT> -L 8501:localhost:8501 root@<IP>
> # Luego: http://localhost:8501
> ```

La UI detecta automáticamente si existe `checkpoints/adapter_config.json` y carga el modelo fine-tuned. El sidebar muestra:
- `LoRA fine-tuned (checkpoints/)` si hay checkpoint
- `Base CLIP (no fine-tuning)` si no

**Modos de búsqueda:**
- **Text Search** — escribe una descripción en lenguaje natural
- **Image Search** — sube una imagen y encuentra las más similares

### CLI (alternativa sin UI)

```bash
# Búsqueda por texto
uv run python scripts/search_cli.py --query "smoke with alpha channel" --top_k 5

# Búsqueda por imagen
uv run python scripts/search_cli.py --image_path data/raw/vfx_assets/depth/img001.jpg

# Modo interactivo mixto
uv run python scripts/search_cli.py
Query > a green screen plate        ← texto
Query > img:/ruta/a/imagen.jpg      ← imagen
```

---

## Función de Pérdida

$$L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j)/\tau)}$$

La temperatura $\tau$ es un parámetro **aprendible** (`nn.Parameter`).

---

## Métricas

| Métrica | Descripción |
|---|---|
| **Recall@1** | % del asset correcto en el top-1 resultado |
| **Train Loss** | Contrastive loss promedio por epoch |

**Baseline (CLIP sin fine-tuning):** Recall@1 = **50%** sobre datos sintéticos.
El objetivo del fine-tuning con LoRA es superar este baseline en el dominio VFX.
