
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transformers import BlipProcessor, BlipForImageTextRetrieval




class BLIPScore(nn.Module):
    def __init__(self, model_name, device='cpu', percentage=True, torch_type=torch.float16):
        super().__init__()
        self.device = device

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(model_name, torch_dtype=torch_type).to(device)
        self.torch_type = torch_type

        self.model.eval()
        self.percentage = percentage

    def score(self, prompts, image_paths):

        if not isinstance(prompts, list):
            prompts = [prompts, ]
            
        if not isinstance(image_paths, list):
            image_paths = [image_paths, ]
            

        scores = []
        
        if len(prompts) == 1 and len(prompts) < len(image_paths):
            prompt = prompts[0]
            for image_path in image_paths:
                raw_image = Image.open(image_path).convert('RGB')
                inputs = self.processor(raw_image, prompt, return_tensors="pt").to(self.device, self.torch_type)
                score = self.model(**inputs, use_itm_head=False)[0]
                if self.percentage:
                    score = score.detach().cpu().item() * 100
                scores.append(score)
        elif len(image_paths) == 1 and len(prompts) > len(image_paths):
            image_path = image_paths[0]
            raw_image = Image.open(image_path).convert('RGB')
            for prompt in prompts:
                inputs = self.processor(raw_image, prompt, return_tensors="pt").to(self.device, self.torch_type)
                score = self.model(**inputs, use_itm_head=False)[0]
                if self.percentage:
                    score = score.detach().cpu().item() * 100
                scores.append(score)
        else:
            for prompt, image_path in zip(prompts, image_paths):
                # score
                raw_image = Image.open(image_path).convert('RGB')
                inputs = self.processor(raw_image, prompt, return_tensors="pt").to(self.device, self.torch_type)
                score = self.model(**inputs, use_itm_head=False)[0]
                if self.percentage:
                    score = score.detach().cpu().item() * 100
                scores.append(score)

        return scores



if __name__ == '__main__':
    config_path = './src/metrics/BLIP/med_config.json'
    device = 'cuda:2'
    blip_score = BLIPScore(med_config=config_path, device=device)
    print(blip_score)
