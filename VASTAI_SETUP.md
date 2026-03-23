# 🚀 vast.ai Setup Guide — ScopeSearch Demo

## 1. Elegir la instancia

En [vast.ai](https://vast.ai), selecciona una instancia con estas características mínimas:

| Parámetro | Mínimo recomendado |
|---|---|
| GPU | RTX 3080 / 3090 / 4080 (16 GB VRAM) |
| RAM | 32 GB |
| Disk | 50 GB SSD |
| Template | **PyTorch 2.x** (preinstalled) |
| Costo aprox. | ~$0.15–$0.40 / hora |

> Buscá en vast.ai con el filtro `pytorch` en templates y ordená por precio. Un RTX 3090 suele estar entre $0.20–$0.35/hr.

---

## 2. Conectarse via SSH

```bash
# En tu terminal local (el comando exacto te lo da vast.ai)
ssh -p <PORT> root@<IP>
```

---

## 3. Setup de entorno en la instancia

```bash
# 1. Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 2. Clonar tu repo (o copiar con scp)
git clone https://github.com/<tu-usuario>/scopesearch.git
cd scopesearch

# 3. Crear entorno y dependencias
uv venv
source .venv/bin/activate
uv pip install torch torchvision transformers peft faiss-cpu pillow pandas tqdm pytest
```

---

## 4. Descargar COCO val 2017 (~1 GB)

```bash
uv run python scripts/download_coco_val.py
```

Descarga automática de:
- `val2017.zip` → 5,000 imágenes
- `annotations_trainval2017.zip` → captions

Genera: `data/processed/coco_val.csv`

---

## 5. Indexar las imágenes (una sola vez, ~2 min en GPU)

```bash
uv run python scripts/index_images.py \
    --metadata data/processed/coco_val.csv \
    --batch_size 128
```

Genera:
- `vector_store/index.faiss` — índice con 5,000 vectores
- `vector_store/image_paths.json` — paths de cada imagen

---

## 6. ¡Buscar!

### One-shot
```bash
uv run python scripts/search_cli.py \
    --query "a dog running on the beach" \
    --top_k 5
```

### Modo interactivo REPL
```bash
uv run python scripts/search_cli.py
```
```
💬 Interactive mode — type a query and press ENTER

🔎 Query > two people cooking in a kitchen
🔍 Query: "two people cooking in a kitchen"
────────────────────────────────────────────────────────────
  #1  [████████████████    ]  0.3421  →  data/raw/coco/val2017/000000123045.jpg
  #2  [████████████████    ]  0.3187  →  data/raw/coco/val2017/000000054321.jpg
  ...
```

---

## 7. Usar tus propios assets VFX (sin CSV)

Si subís una carpeta con tus propias imágenes:

```bash
# Subir desde tu máquina local
scp -P <PORT> -r ./mis_assets root@<IP>:/root/scopesearch/data/raw/

# Indexar sin CSV (detecta JPG/PNG/EXR automáticamente)
uv run python scripts/index_images.py --image_dir data/raw/mis_assets/
```

---

## 8. Tips para vast.ai

- **Apagá la instancia cuando no la uses** — el costo sigue corriendo.
- Montá un **volumen de disco persistente** para no re-descargar el dataset en cada sesión.
- Con GPU activa, el indexado de 5k imágenes COCO tarda ~90 segundos.
- El modelo CLIP se cachea en `~/.cache/huggingface` — la segunda ejecución es instantánea.
