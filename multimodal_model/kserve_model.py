import numpy as np
from typing import Dict
import json
import os
import requests

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

import kserve
from kserve import ModelServer
import logging

class KServeInternVL2(kserve.Model):

    def __init__(self, name: str):
        super().__init__(name)
        KSERVE_LOGGER_NAME = 'kserve'
        self.logger = logging.getLogger(KSERVE_LOGGER_NAME)
        self.name = name
        self.ready = False
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def download_file(self, file_url):
        response = requests.get(file_url)
        if response.status_code == 200:
            with open('processing.jpg', 'wb') as file:
                file.write(response.content)
        return 'processing.jpg'

    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def cleanup_file(self, file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    def load(self):
        # Build tokenizer and model
        name = "OpenGVLab/InternVL2-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
                name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                # load_in_8bit=True,
                use_flash_attn=True,
                trust_remote_code=True).eval().cuda()
        self.ready = True

    def predict(self, request: Dict, headers: Dict) -> Dict:

        data = request
        file_url = data["instances"][0]["inputs"]["image-url"]
        prompt = data["instances"][0]["inputs"]["prompt"]
        self.logger.info(f"file url:-- {file_url}")

        file_path = self.download_file(file_url)
        pixel_values = self.load_image(file_path, max_num=12).to(torch.bfloat16).cuda()

        self.cleanup_file(file_path)

        generation_config = dict(max_new_tokens=1024, do_sample=True)

        response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)

        return {"Name": response}

if __name__ == "__main__":

  model = KServeInternVL2("internvl2")
  model.load()

  model_server = ModelServer(http_port=8080, workers=1)
  model_server.start([model])
