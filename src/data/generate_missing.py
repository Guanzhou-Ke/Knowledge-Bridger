import json
import numpy as np
import random


def simulate_missing_modality(metadata_path, 
                              missing_type, 
                              missing_ratio, 
                              seed=2025,
                              save_path=None):
    """
    Simulate multimodal missing data
    :param data_json: data (json format, including text and image information)
    :param missing_type: missing type ("text", "image", "mixed")
    :param missing_ratio: missing ratio (decimal between 0-1)
    :return: index list of missing data
    """
    
    # 加载数据
    data = json.load(open(metadata_path, 'r'))
    n_samples = len(data)
    missing_count = int(n_samples * missing_ratio)
    
    missing_config = {
        "type": missing_type,
        "missing_ratio": missing_ratio,
    }
    
    # two-element, 1: avaliable, 0: missing.
    # 1st: text, 2nd: image.
    missing_details = {}

    random.seed(seed)
    np.random.seed(seed)

    # Select sample indexes for missing text
    missing_indices = random.sample(range(n_samples), missing_count)
    missing_config['num_missing'] = len(missing_indices)
    missing_config['total'] = n_samples
    missing_config["missing_indices"] = missing_indices
    # Generate missing details.
    if missing_type == "text":
        for idx in missing_indices:
            missing_details[idx] = (0, 1)  # missing text part.
    elif missing_type == "image":
        for idx in missing_indices:
            missing_details[idx] = (1, 0)   # missing Image part.
    elif missing_type == "mixed":
        for idx in missing_indices:
            if random.random() > 0.5:
                missing_details[idx] = (0, 1)  
            else:
                missing_details[idx] = (1, 0)  
    else:
        raise ValueError("Invalid missing_type. Choose from 'text_only', 'image_only', or 'mixed'.")
    
    missing_config["missing_details"] = missing_details
    
    if save_path is not None:
        json.dump(missing_config, open(save_path, 'w'), indent=4)
    
    return missing_config


if __name__ == '__main__':
    import os
    data_path = './src/data/iuxray/trainset.json'
    save_path = './src/data/iuxray/missing-config'
    seed = 2025
    for missing_type in ['text', 'image', 'mixed']:
        for ratio in [0.3, 0.5, 0.7]:
            simulate_missing_modality(data_path, 
                                      missing_type, 
                                      ratio, 
                                      seed=seed,
                                      save_path=os.path.join(save_path, f'train-{missing_type}-{ratio}.json'))
        # different ratio shared same seed.
        seed += 1