from pathlib import Path
from textwrap import dedent
from argparse import ArgumentParser
from os.path import expandvars
from os import system
import sys
import datetime

template = dedent(
    """\
#!/bin/bash
#SBATCH --job-name=mooc-sd
#SBATCH --output={rundir}/output.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition={partition}
#SBATCH --gres={gres}

module load anaconda3 cuda/11.7

cd {codeloc}

python gen.py {gen_args}
"""
)


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def resolve_path(path, to_str=False):
    path = Path(expandvars(str(path))).expanduser().resolve()
    if to_str:
        return str(path)
    return path


def launch_job(args):
    rundir = resolve_path(args.get("rundir"))
    gres = args.pop("gres")
    partition = args.pop("partition")
    codeloc = resolve_path(args.pop("codeloc"))

    gen_args = " ".join([f"--{k}={v}" for k, v in args.items()])

    rundir = resolve_path(args.rundir)
    rundir.mkdir(parents=True, exist_ok=True)

    sbatch_str = template.format(
        partition=partition,
        gres=gres,
        codeloc=codeloc,
        rundir=str(rundir),
        gen_args=gen_args,
    )
    Path("./sbatchs").mkdir(exist_ok=True)
    fname = Path(f"./sbatchs/{now()}.sh").resolve()
    fname.write_text(sbatch_str)
    system(f"sbatch {str(fname)}")
    print(f"Launched job from {str(fname)}")


def save_im(im, name, prompt, base_dir="./mooc-images/"):
    o = Path(base_dir)
    o.mkdir(exist_ok=True, parents=True)
    im_out = o / f"{name}.png"
    prompt_out = o / f"{name}.txt"
    im.save(str(im_out))
    prompt_out.write_text(prompt)
    print("Saved", str(im_out), str(prompt_out))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--partition", type=str, default="main")
    parser.add_argument("--gres", type=str, default="gpu:rtx8000:1")
    parser.add_argument("--codeloc", type=str, default=".")
    parser.add_argument("--rundir", type=str, default="$SCRATCH/mooc/sd")
    parser.add_argument("--launch", action="store_true", default=False)
    parser.add_argument("--prompt_file", type=str, default="prompts/victor.yaml")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=80)
    parser.add_argument("--guidance_scale", type=float, default=6.5)
    parser.add_argument("--num_images_per_prompt", type=int, default=8)
    args = dict(vars(parser.parse_args()))

    launch = args.pop("launch")
    if launch:
        launch_job(args)
        sys.exit(0)

    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    import torch
    import yaml

    model_id = "stabilityai/stable-diffusion-2-1"
    rundir = resolve_path(args["rundir"])

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    prompts = yaml.safe_load(resolve_path(args["prompt_file"]).read_text())

    for name, prompt in prompts.items():
        ims = pipe(
            prompt,
            height=1024,
            width=1024,
            num_inference_steps=80,
            guidance_scale=6.5,
            num_images_per_prompt=8,
        ).images
        print("Generating:", prompt)
        (rundir / name / "prompt.txt").write_text(prompt)
        for i, im in enumerate(ims):
            print(i, end="\r")
            save_im(im, f"{name}/{i}", prompt, base_dir=rundir)
