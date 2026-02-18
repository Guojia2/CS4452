import modal
from modal_pipeline.app import app, image, volume, VOLUME_MOUNT_PATH
import os

@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=3600,  # 1 hour — uploading video data takes time
)
def upload_dataset(local_data_path: str):
    """
    Call this locally once to push your THUMOS data into the Modal Volume.
    Usage:  modal run modal_pipeline/data_prep.py
    """
    import shutil
    dest = os.path.join(VOLUME_MOUNT_PATH, "thumos")
    print(f"Copying data to {dest}...")
    shutil.copytree(local_data_path, dest, dirs_exist_ok=True)
    volume.commit()  # Flush writes to the Volume
    print("Done.")


@app.local_entrypoint()
def main():
    # Change this to wherever your THUMOS data lives locally
    upload_dataset.remote("/path/to/local/thumos")