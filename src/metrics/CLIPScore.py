import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import clip


class CLIPScore(nn.Module):
    def __init__(self, download_root, device='cpu', percentage=True):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device, jit=False, 
                                                     download_root=download_root)
        
        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(self.clip_model) # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.clip_model.eval()
        self.clip_model.logit_scale.requires_grad_(False)
        self.percentage = percentage

    def score(self, prompts, image_paths):
        
        if not isinstance(prompts, list):
            prompts = [prompts, ]
            
        if not isinstance(image_paths, list):
            image_paths = [image_paths, ]
            
        txt_features = []    
        for prompt in prompts:
            # text encode
            text = clip.tokenize(prompt, truncate=True).to(self.device)
            txt_feature = F.normalize(self.clip_model.encode_text(text))
            txt_features.append(txt_feature)
            
        image_features = []
        # image encode
        for image_path in image_paths:
            pil_image = Image.open(image_path)
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_feature = F.normalize(self.clip_model.encode_image(image))
            image_features.append(image_feature)

        scores = []
        if len(txt_features) == 1 and len(txt_features) < len(image_features):
            txt_feature = txt_features[0]
            for image_feature in image_features:
                score = max(torch.sum(torch.mul(txt_feature, image_feature), dim=1, keepdim=True), torch.tensor(0))
                if self.percentage:
                    score = score.detach().cpu().item() * 100
                scores.append(score)
        elif len(image_features) == 1 and len(txt_features) > len(image_features):
            image_feature = image_features[0]
            for txt_feature in txt_features:
                # score
                # score = max((image_features * txt_feature).sum(axis=1), torch.tensor(0))
                score = max(torch.sum(torch.mul(txt_feature, image_feature), dim=1, keepdim=True), torch.tensor(0))
                if self.percentage:
                    score = score.detach().cpu().item() * 100
                scores.append(score)
        else:
            for txt_feature, image_feature in zip(txt_features, image_features):
                # score
                # score = max((image_features * txt_feature).sum(axis=1), torch.tensor(0))
                score = max(torch.sum(torch.mul(txt_feature, image_feature), dim=1, keepdim=True), torch.tensor(0))
                if self.percentage:
                    score = score.detach().cpu().item() * 100
                scores.append(score)
        return scores

    def score_same_modality(self, m1, m2, modality='text'):
        """
        Computing the same modality's similarity score (SS in the paper.).
        """
        features = [] 
        if modality == 'text':
            for prompt in [m1, m2]:
                # text encode
                text = clip.tokenize(prompt, truncate=True).to(self.device)
                txt_feature = F.normalize(self.clip_model.encode_text(text))
                features.append(txt_feature)
        elif modality == 'image':
            for image_path in [m1, m2]:
                pil_image = Image.open(image_path)
                image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                image_features = F.normalize(self.clip_model.encode_image(image))
                features.append(image_features)
                
        score = max(torch.sum(torch.mul(features[0], features[1]), dim=1, keepdim=True), torch.tensor(0))
        return score.detach().cpu().item()

    def inference_rank(self, prompt, generations_list):
        
        text = clip.tokenize(prompt, truncate=True).to(self.device)
        txt_feature = F.normalize(self.clip_model.encode_text(text))
        
        txt_set = []
        img_set = []
        for generations in generations_list:
            # image encode
            img_path = generations
            pil_image = Image.open(img_path)
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_features = F.normalize(self.clip_model.encode_image(image))
            img_set.append(image_features)
            txt_set.append(txt_feature)
            
        txt_features = torch.cat(txt_set, 0).float() # [image_num, feature_dim]
        img_features = torch.cat(img_set, 0).float() # [image_num, feature_dim]
        rewards = torch.sum(torch.mul(txt_features, img_features), dim=1, keepdim=True)
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1
        
        return indices.detach().cpu().numpy().tolist(), rewards.detach().cpu().numpy().tolist()