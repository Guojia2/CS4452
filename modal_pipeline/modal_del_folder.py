import modal

app = modal.App("inline-del-pipeline")

image = (   
    modal.Image.debian_slim(python_version="3.11")
    .add_local_dir(".", remote_path="/root/project")
)

thumos14 = modal.Volume.from_name("Thumos14", create_if_missing=False)

@app.function(
    image=image,
    volumes={"/vol": thumos14},
)
def delete_path(path: str):
    import os
    import shutil

    full_path = os.path.join("/vol", path)

    if os.path.exists(full_path):
        shutil.rmtree(full_path)
        print(f"Deleted: {full_path}")
    else:
        print(f"Path does not exist: {full_path}")

    thumos14.commit()

@app.local_entrypoint()
def main():
    delete_path.remote("features/videomae2-new")