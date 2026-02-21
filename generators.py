import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from PIL import Image
import time
import os

# Глобальные переменные для однократной загрузки моделей
_pipe_gpu = None
_pipe_cpu = None

def get_gpu_pipe():
    global _pipe_gpu
    if _pipe_gpu is None:
        print("Загрузка GPU модели...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            safety_checker=None,  # отключаем для скорости
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        _pipe_gpu = pipe
    return _pipe_gpu

def get_cpu_pipe():
    global _pipe_cpu
    if _pipe_cpu is None:
        print("Загрузка CPU модели (Small Stable Diffusion)...")
        pipe = DiffusionPipeline.from_pretrained(
            "OFA-Sys/small-stable-diffusion-v0",
            safety_checker=None,
            torch_dtype=torch.float32
        )
        pipe.to("cpu")
        _pipe_cpu = pipe
    return _pipe_cpu

def generate_gpu(prompt, output_path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA не доступна, невозможно использовать GPU генератор.")
    pipe = get_gpu_pipe()
    print("Генерация на GPU...")
    start = time.time()
    # Стандартные параметры (512x512, 50 шагов)
    image = pipe(prompt).images[0]
    image.save(output_path)
    print(f"Генерация на GPU завершена за {time.time()-start:.1f} сек")
    return output_path

def generate_cpu(prompt, output_path):
    pipe = get_cpu_pipe()
    print("Генерация на CPU...")
    start = time.time()
    # Уменьшенные параметры для скорости
    image = pipe(
        prompt,
        num_inference_steps=10,
        height=256,
        width=256,
        guidance_scale=6.0
    ).images[0]
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    image.save(output_path)
    print(f"Генерация на CPU завершена за {time.time()-start:.1f} сек")
    return output_path