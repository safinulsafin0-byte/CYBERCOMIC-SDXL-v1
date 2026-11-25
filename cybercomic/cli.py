import argparse
import os
import time
import torch
import peft.import_utils
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

# Windows symlink ওয়ার্নিং বন্ধ এবং পুরনো torchao থাকলে ক্র্যাশ এড়ানোর সেফটি প্যাচ
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
peft.import_utils.is_torchao_available = lambda: False

DEFAULT_NEG = (
    "speech bubble, text, caption box, watermark, "
    "3d render, photorealistic, blurry, low quality, deformed hands"
)


def load_pipeline(offline: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[CYBERCOMIC] Loading model on {device.upper()} ({dtype})...")

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=dtype,
        local_files_only=offline,
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
        local_files_only=offline,
    ).to(device)

    pipe.enable_attention_slicing()
    pipe.load_lora_weights(
        "Safinul/CYBERCOMIC-SDXL-v1",
        weight_name="cybercomic_sdxl_v1.safetensors",
        local_files_only=offline,
    )
    return pipe, device


def generate_image(pipe, device, prompt, neg, output, steps, scale, width, height, seed):
    full_prompt = prompt.strip()
    if "cb_comic_style" not in full_prompt:
        full_prompt = f"cb_comic_style, comic book character art, {full_prompt}"

    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    print(f"[CYBERCOMIC] Generating: {full_prompt}")
    image = pipe(
        prompt=full_prompt,
        negative_prompt=neg,
        num_inference_steps=steps,
        guidance_scale=7.0,
        width=width,
        height=height,
        generator=generator,
        cross_attention_kwargs={"scale": scale},
    ).images[0]

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    image.save(output)
    print(f"[CYBERCOMIC] Saved -> {output}")


def main():
    parser = argparse.ArgumentParser(
        prog="cybercomic",
        description="Generate high-contrast sci-fi & superhero comic art from your terminal.",
    )
    parser.add_argument("prompt", nargs="?", help="Scene or character prompt (omit for interactive mode)")
    parser.add_argument("-o", "--output", default="comic_panel.png", help="Output image path")
    parser.add_argument("-n", "--neg", default=DEFAULT_NEG, help="Negative prompt")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps (default: 25)")
    parser.add_argument("--scale", type=float, default=0.9, help="LoRA strength 0.7-1.0 (default: 0.9)")
    parser.add_argument("--width", type=int, default=768, help="Image width (default: 768)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--offline", action="store_true", help="Run 100%% offline from local HF cache")
    args = parser.parse_args()

    pipe, device = load_pipeline(offline=args.offline)

    if args.prompt:
        generate_image(
            pipe, device, args.prompt, args.neg, args.output,
            args.steps, args.scale, args.width, args.height, args.seed
        )
    else:
        print("\n[CYBERCOMIC] Interactive Mode — type a prompt and press Enter (type 'exit' to quit)\n")
        while True:
            try:
                user_prompt = input("Prompt > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_prompt or user_prompt.lower() in {"exit", "quit", "q"}:
                break
            out_file = f"comic_{int(time.time())}.png"
            generate_image(
                pipe, device, user_prompt, args.neg, out_file,
                args.steps, args.scale, args.width, args.height, args.seed
            )


if __name__ == "__main__":
    main()