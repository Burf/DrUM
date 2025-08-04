import os
import numpy as np
import torch
import argparse
from drum import DrUM
from diffusers import DiffusionPipeline, StableDiffusion3Pipeline, FluxPipeline

def main():
    parser = argparse.ArgumentParser(description="Draw Your Mind: Personalized Generation via Condition-Level Modeling in Text-to-Image Diffusion Models")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="T2I models id")
    parser.add_argument("--sampling", type=bool, default=False, help="Enable or disable sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--prompt", type=str, default="a photograph of an astronaut riding a horse", help="Prompt for image generation")
    parser.add_argument("--ref", nargs='+', default=["A retro-futuristic space exploration movie poster with bold, vibrant colors"], help="Reference descriptions for conditioning")
    parser.add_argument("--weight", nargs='+', type=float, default=[1], help="Weight for the reference descriptions")
    parser.add_argument("--alpha", type=float, default=0.3, help="Alpha scaling factor (range: 0-1)")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Number of images per prompt")
    parser.add_argument("--save_path", type=str, default="./image", help="Directory to save generated images")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID to use")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_seed(args.seed)
    dtype = torch.bfloat16

    # ============================
    # Load DrUM pipeline
    # ============================
    if "flux" in args.model.lower():
        pipeline = FluxPipeline.from_pretrained(args.model, torch_dtype = dtype)
    elif "stable-diffusion-3.5" in args.model.lower():
        pipeline = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype = dtype)
    else:
        pipeline = DiffusionPipeline.from_pretrained(args.model, torch_dtype = dtype)
    pipeline = pipeline.to("cuda")
    #pipeline.safety_checker = lambda images, clip_input: (images, [False] * len(images))

    #if you want to use manual weights, you can specify the weights directory, for example: "weight = ./weight"
    #drum = DrUM(args.model, torch_dtype = dtype, device = "cuda")
    drum = DrUM(pipeline)

    # ============================
    # Personalized generation in T2I diffusion models
    # ============================
    image = drum(args.prompt, args.ref, weight = args.weight, alpha = args.alpha, sampling = args.sampling, skip = None, seed = args.seed,
                 size = None, num_inference_steps = None, num_images_per_prompt = args.num_images_per_prompt)

    # ============================
    # Save Results
    # ============================
    os.makedirs(args.save_path, exist_ok=True)
    for i, img in enumerate(image):
        img.save(os.path.join(args.save_path, f"{i:05d}.png"))

    print(f"Inference complete. Images saved to {args.save_path}")

if __name__ == "__main__":
    main()