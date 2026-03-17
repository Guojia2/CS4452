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
        "raw/videos/val":  "https://drive.google.com/drive/folders/1s-vldTTvgF0Dn8x3aoMuB7UkMVkv5NNU",
        "raw/videos/test": "https://drive.google.com/drive/folders/13JbXBoKJu8Me8jjAR0mCdvADLZbs-1LM>",
        "raw/annotations": "https://drive.google.com/drive/folders/1K8l4CzxBVgo9Y6u2Iu0YbKAa2IShnzjo",
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