import json
from pathlib import Path

import torch
import clip
from torch.utils.data import Dataset
from PIL import Image
import random



class COCOMultiLabelDataset(Dataset):
    def __init__(self, 
                 root: Path, 
                 clip_download_root: Path,
                 class_name_path: Path, 
                 metadata_path: Path,
                 seed=2025,
                 isMissing=False,
                 missing_config: Path=None,
                 keep_raw=False,
                 merge_metadata=False,
                 ):
        self.root = root
        self.class_name = json.load(open(class_name_path, 'r'))
        self.metadata = json.load(open(metadata_path, 'r'))
        self.isMissing = isMissing
        if self.isMissing:
            self.missing_config = json.load(open(missing_config, 'r'))
        self.keep_raw = keep_raw
        self.merge_metadata = merge_metadata
        self.num_classes = len(self.class_name['label_name'])
        
        if not self.keep_raw:
            _, self.img_preprocess = clip.load("ViT-L/14", 
                                            device='cpu', 
                                            download_root=clip_download_root, 
                                            jit=False)
        
        self.seed = seed
        # set random seed.
        random.seed(self.seed)
        
        # merge the missing indices into original metadata.
        if self.isMissing and self.merge_metadata:
            new_metadata = []
            for idx, item in enumerate(self.metadata):
                if idx not in self.missing_config['missing_indices']:
                    new_metadata.append(item)
            self.metadata = new_metadata
            
        

    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, idx):
        image_path = self.root / self.metadata[idx]['file_name']
        # random choice a caption to increase the diversity.
        # random_text = random.choice(self.metadata[idx]['captions'])
        random_text = self.metadata[idx]['captions'][0]
        
        label_idx = self.metadata[idx]['labels']
        
        
        # training in missing.
        if self.isMissing and (not self.merge_metadata):
            if idx in self.missing_config['missing_indices']:
                md = self.missing_config['missing_details'][f"{idx}"]
                if md[1] == 0:
                    for idx in range(len(image)):
                        image[idx] = torch.ones(image[idx].size()).float()
                #missing text, dummy text is '' 
                elif md[0] == 0:
                    random_text = ''
        
        label = torch.zeros(self.num_classes)
        label[label_idx] = 1.
        image = self.img_preprocess(Image.open(image_path))
        text = clip.tokenize([random_text])[0]  # get text toekn.
        
        return image, text, label


class MMIMDBMultiLabelDataset(Dataset):
    def __init__(self, 
                 root: Path, 
                 clip_download_root: Path,
                 class_name_path: Path, 
                 metadata_path: Path,
                 seed=2025,
                 isMissing=False,
                 missing_config: Path=None,
                 keep_raw=False,
                 merge_metadata=False,
                 ):
        self.root = root
        self.class_name = json.load(open(class_name_path, 'r'))
        self.metadata = json.load(open(metadata_path, 'r'))
        self.isMissing = isMissing
        if self.isMissing:
            self.missing_config = json.load(open(missing_config, 'r'))
        self.keep_raw = keep_raw
        self.merge_metadata = merge_metadata
        self.num_classes = len(self.class_name['label_name'])
        
        if not self.keep_raw:
            _, self.img_preprocess = clip.load("ViT-L/14", 
                                            device='cpu', 
                                            download_root=clip_download_root, 
                                            jit=False)
        
        self.seed = seed
        # set random seed.
        random.seed(self.seed)
        
        # merge the missing indices into original metadata.
        if self.isMissing and self.merge_metadata:
            new_metadata = []
            for idx, item in enumerate(self.metadata):
                if idx not in self.missing_config['missing_indices']:
                    new_metadata.append(item)
            self.metadata = new_metadata
            
        

    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, idx):
        image_path = self.root / self.metadata[idx]['file_name']
        # It would be long text.
        plot = ' '.join(self.metadata[idx]['captions'])
        
        label_name = self.metadata[idx]['labels']
        label_idx = [self.class_name['label2id'][name] for name in label_name]
        
        label = torch.zeros(self.num_classes)
        label[label_idx] = 1.
        image = self.img_preprocess(Image.open(image_path))
        
        # training in missing.
        if self.isMissing and (not self.merge_metadata):
            if idx in self.missing_config['missing_indices']:
                md = self.missing_config['missing_details'][f"{idx}"]
                if md[1] == 0:
                    for idx in range(len(image)):
                        image[idx] = torch.ones(image[idx].size()).float()
                #missing text, dummy text is '' 
                elif md[0] == 0:
                    plot = ''
        
        text = clip.tokenize([plot], truncate=True)[0]  # get text toekn.
        
        return image, text, label
    
    
class IUXRAYMultiLabelDataset(Dataset):
    def __init__(self, 
                 root: Path, 
                 clip_download_root: Path,
                 class_name_path: Path, 
                 metadata_path: Path,
                 seed=2025,
                 isMissing=False,
                 missing_config: Path=None,
                 keep_raw=False,
                 merge_metadata=False,
                 ):
        self.root = root
        self.class_name = json.load(open(class_name_path, 'r'))
        self.metadata = json.load(open(metadata_path, 'r'))
        self.isMissing = isMissing
        if self.isMissing:
            self.missing_config = json.load(open(missing_config, 'r'))
        self.keep_raw = keep_raw
        self.merge_metadata = merge_metadata
        self.num_classes = len(self.class_name['label_name'])
        
        if not self.keep_raw:
            _, self.img_preprocess = clip.load("ViT-L/14", 
                                            device='cpu', 
                                            download_root=clip_download_root, 
                                            jit=False)
        
        self.seed = seed
        # set random seed.
        random.seed(self.seed)
        
        # merge the missing indices into original metadata.
        if self.isMissing and self.merge_metadata:
            new_metadata = []
            for idx, item in enumerate(self.metadata):
                if idx not in self.missing_config['missing_indices']:
                    new_metadata.append(item)
            self.metadata = new_metadata
            
        
            
        

    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, idx):
        image_path = self.root / self.metadata[idx]['file_name']
        # It would be long text.
        findings = self.metadata[idx]['captions'][0]
        
        label_name = self.metadata[idx]['labels']
        label_idx = [self.class_name['label2id'][name] for name in label_name]
        
        # training in missing.
        if self.isMissing and (not self.merge_metadata):
            if idx in self.missing_config['missing_indices']:
                md = self.missing_config['missing_details'][f"{idx}"]
                if md[1] == 0:
                    for idx in range(len(image)):
                        image[idx] = torch.ones(image[idx].size()).float()
                #missing text, dummy text is '' 
                elif md[0] == 0:
                    findings = ''
        
        label = torch.zeros(self.num_classes)
        label[label_idx] = 1.
        image = self.img_preprocess(Image.open(image_path))
        text = clip.tokenize([findings], truncate=True)[0]  # get text toekn.
        
        return image, text, label


if __name__ == '__main__':
    # val_root= '[your path]/COCO2014/val2014/'
    # clip_download_root= 'clip_models'
    # class_name_path= './src/data/mscoco/category.json'
    # test_metadata_path= './src/data/mscoco/train_anno.json'
    # missing_config_path = './src/data/mscoco/missing-config/train-text-0.7.json'
    
    
    # dataset = COCOMultiLabelDataset(val_root,
    #                       clip_download_root,
    #                       class_name_path,
    #                       test_metadata_path,
    #                       isMissing=True,
    #                       missing_config=missing_config_path,
    #                       keep_raw=True,
    #                       merge_metadata=True)
    
    # print(len(dataset))
    
    val_root= '[your path]/mmimdb/mmimdb/dataset/'
    clip_download_root= 'clip_models'
    class_name_path= './src/data/mmimdb/category.json'
    test_metadata_path= './src/data/mmimdb/trainset.json'
    missing_config_path = './src/data/mmimdb/missing-config/train-mixed-0.7.json'
    
    
    dataset = MMIMDBMultiLabelDataset(Path(val_root),
                          clip_download_root,
                          class_name_path,
                          test_metadata_path,
                          isMissing=True,
                          missing_config=missing_config_path,
                          keep_raw=False,
                          merge_metadata=False)
    
    print(len(dataset))
    print(dataset[1])
    
    
    # val_root= '[your path]/IU-XRay/images'
    # clip_download_root= 'clip_models'
    # class_name_path= './src/data/iuxray/category.json'
    # test_metadata_path= './src/data/iuxray/trainset.json'
    # missing_config_path = './src/data/iuxray/missing-config/train-text-0.7.json'
    
    
    # dataset = IUXRAYMultiLabelDataset(Path(val_root),
    #                       clip_download_root,
    #                       class_name_path,
    #                       test_metadata_path,
    #                       isMissing=False,
    #                     #   missing_config=missing_config_path,
    #                       keep_raw=False,
    #                       merge_metadata=False)
    
    # print(len(dataset))
    # print(dataset[0])
    
    
    