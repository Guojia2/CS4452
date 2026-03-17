import modal
import os
from modal_pipeline.app import app, image, volume, VOLUME_MOUNT_PATH

download_image = image.pip_install("gdown")

@app.function(
    image=download_image,
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=60 * 60 * 12,
)
def download_from_gdrive():
    import gdown

    splits = {
        "raw/videos/val":  "https://drive.google.com/drive/folders/<val_folder_id>",
        "raw/videos/test": "https://drive.google.com/drive/folders/<test_folder_id>",
        "raw/annotations": "https://drive.google.com/drive/folders/<annotations_folder_id>",
    }

    for dest_subpath, url in splits.items():
        dest = os.path.join(VOLUME_MOUNT_PATH, dest_subpath)
        os.makedirs(dest, exist_ok=True)
        print(f"Downloading {dest_subpath}...")
        gdown.download_folder(url=url, output=dest, quiet=False, resume=True)

    volume.commit()
    print("All done — data committed to Modal Volume.")

@app.local_entrypoint()
def main():
    download_from_gdrive.remote()