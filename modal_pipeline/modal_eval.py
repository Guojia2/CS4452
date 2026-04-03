import modal

app = modal.App("inline-eval-pipeline")

thumos14 = modal.Volume.from_name("Thumos14", create_if_missing=False)

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


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 12,
    volumes={"/vol": thumos14},
)
def eval_remote(
    config_path: str,
    ckpt_path: str,
    epoch: int = -1,
    topk: int = -1,
    saveonly: bool = False,
    print_freq: int = 10,
):
    import os
    import sys

    os.chdir("/root/project")
    sys.path.insert(0, "/root/project")
    sys.path.insert(0, "/root/project/libs/utils")

    from eval import main

    class Args:
        def __init__(self):
            self.config = config_path
            self.ckpt = ckpt_path
            self.epoch = epoch
            self.topk = topk
            self.saveonly = saveonly
            self.print_freq = print_freq

    main(Args())
    thumos14.commit()


@app.local_entrypoint()
def run(
    config_path: str,
    ckpt_path: str,
    epoch: int = -1,
    topk: int = -1,
    saveonly: bool = False,
    print_freq: int = 10,
):
    eval_remote.remote(
        config_path=config_path,
        ckpt_path=ckpt_path,
        epoch=epoch,
        topk=topk,
        saveonly=saveonly,
        print_freq=print_freq,
    )