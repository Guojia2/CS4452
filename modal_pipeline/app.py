import modal

app = modal.App("thumos-action-recognition")

# ── Volumes ──────────────────────────────────────────────────────────────────
# Dataset lives in the shared 'main' environment — read-only for us.
# Never write to this volume.
dataset_volume = modal.Volume.from_name("thumos-vol", environment_name="main")
DATASET_PATH = "/data"   # raw videos + annotations mount point

# Our working volume lives in the active environment (alex-dev by default).
# Features, checkpoints, and logs are written here.
work_volume = modal.Volume.from_name("alex-work", create_if_missing=True)
WORK_PATH = "/work"      # features, checkpoints, logs mount point

# ── Container Image ───────────────────────────────────────────────────────────
# Built once and cached. Rebuild only happens when this definition changes.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        extra_index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install(
        "pytorchvideo==0.1.5",
        "av==16.1.0",
        "decord",       # fast batch frame fetching — primary decoder
        "Pillow",
        "numpy<2",      # torch 2.1.2 requires numpy 1.x; numpy 2.x breaks it
        "PyYAML",
        "tqdm",
        "fvcore",
        "iopath",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("modal_pipeline", remote_path="/root/modal_pipeline")
)

GPU = "a10g"   # 24 GB VRAM, good balance of speed and cost (~$1.10/hr)
