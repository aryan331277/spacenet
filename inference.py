from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from PIL import Image
import cv2
import numpy as np
import os

def load_controlnet_pipeline():
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

def preprocess_sketch(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, img_bin = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
    img_bin = cv2.GaussianBlur(img_bin, (5,5), 0)
    edge_image = cv2.Canny(img_bin, 100, 200)
    edge_image = Image.fromarray(edge_image).convert("RGB")
    return edge_image

def generate_city(pipe, prompt="satellite view of a modern futuristic city", sketch=None):
    generated_image = pipe(prompt=prompt, image=sketch).images[0]
    return generated_image
