import modal
import os

app = modal.App("thumos-action-recognition")
volume = modal.Volume.from_name("thumos-vol", create_if_missing=True)
VOLUME_MOUNT_PATH = "/vol"

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("rclone")
)

rclone_secret = modal.Secret.from_name("rclone-config")

@app.function(
    image=download_image,
    volumes={VOLUME_MOUNT_PATH: volume},
    secrets=[rclone_secret],
    timeout=60 * 60 * 12,
)
def download_from_gdrive():
    import subprocess
    import os

    config_path = "/root/.config/rclone/rclone.conf"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        f.write(os.environ["RCLONE_CONFIG_CONTENT"])

    splits = {
        "raw/videos/val":  "1s-vldTTvgF0Dn8x3aoMuB7UkMVkv5NNU",
        "raw/videos/test": "13JbXBoKJu8Me8jjAR0mCdvADLZbs-1LM",
        "raw/annotations": "1K8l4CzxBVgo9Y6u2Iu0YbKAa2IShnzjo",
    }

    for dest_subpath, folder_id in splits.items():
        dest = os.path.join(VOLUME_MOUNT_PATH, dest_subpath)
        os.makedirs(dest, exist_ok=True)
        print(f"Downloading {dest_subpath}...")
        result = subprocess.run([
            "rclone", "copy",
            "gdrive:",
            dest,
            "--config", config_path,
            "--drive-root-folder-id", folder_id,
            "--progress",
            "--transfers", "8",
        ], capture_output=True, text=True)

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        if result.returncode != 0:
            raise RuntimeError(f"rclone failed: {result.stderr}")
        volume.commit()
        print(f"Committed {dest_subpath}.")

    print("All done.")

@app.local_entrypoint()
def main():
    download_from_gdrive.remote()