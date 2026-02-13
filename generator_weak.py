import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def generate_image_fast_quantized(
    prompt="A beautiful tree in a sunny forest",
    output_path="tree.png",
    num_inference_steps=15,        # ещё меньше шагов
    height=128,
    width=128
):
    # Загружаем лёгкую модель
    model_id = "OFA-Sys/small-stable-diffusion-v0"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        safety_checker=None,
        torch_dtype=torch.float32
    )

    # Устанавливаем быстрый планировщик
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Применяем динамическое квантование к UNet и VAE (для CPU)
    pipe.unet = torch.quantization.quantize_dynamic(
        pipe.unet, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )
    pipe.vae = torch.quantization.quantize_dynamic(
        pipe.vae, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )

    device = "cpu"  # принудительно CPU (квантование эффективно только на CPU)
    pipe.to(device)

    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width
    ).images[0]

    image.save(output_path)
    print(f"Изображение сохранено как {output_path} ({height}x{width})")

if __name__ == "__main__":
    generate_image_fast_quantized()