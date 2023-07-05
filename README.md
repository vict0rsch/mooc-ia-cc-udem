# mooc-ia-cc-udem

Example:

1/ Make a yaml file in `prompts` like:

```yaml
# in prompts/victor.yaml
# name: prompt

renewables: A 4K picure of variable renewable energy sources
new-materials: A 4K illustration of a laboratory testing new materials
electricity-network: A 4K picture of an electricity grid, viewed from above, in the forest
satellite: An HD satellite image of the Pacific Ocean viewed from space
carbon-capture: A 4K picture of a carbon capture system on a city roof
```

2/ Launch a job pointing to that yaml file:

```bash
python gen.py --launch --prompt_file=./prompts/victor.yaml
```

This will:

1. Write an `sbatch` file in `./sbatchs/` with the date of the command
2. Submit the file to the SLURM scheduler
3. Generate the images in `args.rundir / {prompt_name} / {i}.png`
   1. `args.rundir` defaults to `$SCRATCH/mooc/sd`

Parser params:

```python
parser.add_argument("--partition", type=str, default="main")
parser.add_argument("--gres", type=str, default="gpu:rtx8000:1")
parser.add_argument("--codeloc", type=str, default=".")
parser.add_argument("--rundir", type=str, default="$SCRATCH/mooc/sd")
parser.add_argument("--env", type=str, default="stable-diffusion-2.1")
parser.add_argument("--launch", action="store_true", default=False)
parser.add_argument("--prompt_file", type=str, default="prompts/victor.yaml")
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--num_inference_steps", type=int, default=80)
parser.add_argument("--guidance_scale", type=float, default=6.5)
parser.add_argument("--num_images_per_prompt", type=int, default=8)
```
