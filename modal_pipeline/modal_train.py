import modal

app = modal.App("inline-training-pipeline")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("build-essential", "gcc", "g++")
    .pip_install(
        "torch",
        "torchvision",
        "tensorboard",
        "pyyaml",
        "h5py",
        "pandas",
        "joblib",
        "numpy",
        "tqdm",
        "easydict",
        "transformers==4.30.2",
        "einops",
        "modal",
        "timm"

    )
    .add_local_dir(".", remote_path="/root/project", copy=True)
    .run_commands(
        "cd /root/project/libs/utils && python setup.py build_ext --inplace"
    )
)

thumos14 = modal.Volume.from_name("Thumos14", create_if_missing=False)

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 12,
    volumes={"/vol": thumos14},
)
def train_remote(
    config_path: str,
    print_freq: int = 10,
    ckpt_freq: int = 5,
    output: str = "",
    resume: str = "",
):
    import os
    import sys

    os.chdir("/root/project")
    sys.path.insert(0, "/root/project")
    sys.path.insert(0, "/root/project/libs/utils")

    print("exists /vol:", os.path.exists("/vol"))
    print("exists /vol/annotations:", os.path.exists("/vol/annotations"))
    print("exists json:", os.path.exists("/vol/annotations/thumos14_af.json"))
    print("exists feat dir:", os.path.exists("/vol/features/pretrained/i3d_features"))

    if os.path.exists("/vol"):
        print("/vol:", os.listdir("/vol"))

    if os.path.exists("/vol/annotations"):
        print("/vol/annotations:", os.listdir("/vol/annotations"))

    if os.path.exists("/vol/features/pretrained"):
        print("/vol/features/pretrained:", os.listdir("/vol/features/pretrained"))

    from train import main

    class Args:
        def __init__(self):
            self.config = config_path
            self.print_freq = print_freq
            self.ckpt_freq = ckpt_freq
            self.output = output
            self.resume = resume
            self.start_epoch = 0

    main(Args())
    thumos14.commit()


@app.local_entrypoint()
def run(
    config_path: str,
    print_freq: int = 10,
    ckpt_freq: int = 5,
    output: str = "",
    resume: str = "",
):
    train_remote.remote(
        config_path=config_path,
        print_freq=print_freq,
        ckpt_freq=ckpt_freq,
        output=output,
        resume=resume,
    )