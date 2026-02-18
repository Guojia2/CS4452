import modal

# --- App ---
app = modal.App("thumos-action-recognition")

# --- Persistent Volume (survives across runs) ---
# Use this for your dataset and checkpoints so you don't re-upload every time
volume = modal.Volume.from_name("thumos-vol", create_if_missing=True)
VOLUME_MOUNT_PATH = "/vol"  # Where the volume appears inside the container

# --- Container Image ---
# Modal builds this once and caches it
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    # If you need ffmpeg for video decoding:
    .apt_install("ffmpeg")
)