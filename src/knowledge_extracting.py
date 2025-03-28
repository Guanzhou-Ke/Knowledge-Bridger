import argparse
import os

from tqdm import tqdm

from utils import (load_config, 
                   reproducibility_setting,
                   load_json_from_file, 
                   save_json)
from misc.mylogging import get_logger
from knowledge_bridger.large_model_agents import LMAgent
from knowledge_bridger import (GENERAL_ELEMENTS, 
                               GENERAL_ELEMENTS_RETURN, 
                               KnowledgeExtractingProto)
from knowledge_bridger import  (MedicalKnowledgeExtractionProto,
                                CHEST_XRAY_EXAMPLES, 
                                MEDICAL_KNOWLEDGE_RETURN)


def parse_args():
    parser = argparse.ArgumentParser(prog='Knowledge extraction.',)
    
    parser.add_argument('--config', '-f', help='Config file path')
    
    args = parser.parse_args()
    return args


def get_missing_dataset(config):
    """
    "input_type": "text",
    "text_content": item['captions'][0],
    "image_content": None,
    "GT_text": item['captions'],
    "GT_image": [root / item['file_name']],
    """
    missing_config = load_json_from_file(config.missing_config)
    metadata = load_json_from_file(config.train_metadata_path)
    text_ops = 'join' if config.name == 'mmimdb' else 'first'
    text_num_missing = 0
    image_num_missing = 0
    missing_dataset = []
    for idx in missing_config['missing_indices']:
        missing_type = 'text' if missing_config['missing_details'][str(idx)][0] == 0 else 'image'
        input_type = 'text' if missing_config['missing_details'][str(idx)][0] == 1 else 'image'
        if missing_type == 'image':
            image_num_missing += 1
            if text_ops == 'join':
                text_content = ' '.join(metadata[idx]['captions'])
                GT_text = ' '.join(metadata[idx]['captions'])
            else:
                text_content = metadata[idx]['captions'][0]
                GT_text = metadata[idx]['captions'][0]
            image_content = None
        else:
            text_num_missing += 1
            text_content = None
            GT_text = metadata[idx]['captions'][0]
            image_content = [os.path.join(config.train_root, metadata[idx]['file_name'])]
        GT_image = os.path.join(config.train_root, metadata[idx]['file_name'])
        
        missing_dataset.append({
            "missing_idx": idx,
            "missing_type": missing_type,
            "input_type": input_type,
            "text_content": text_content,
            "image_content": image_content,
            "GT_text": GT_text,
            "GT_image": GT_image
        })
        
    return missing_dataset, text_num_missing, image_num_missing


def knowledge_modeling(proto, missing_dataset, logger, config):
    
    if config.domain == 'general':
        domain_knowledge = GENERAL_ELEMENTS
        return_format = GENERAL_ELEMENTS_RETURN
    elif config.domain == 'medical':
        domain_knowledge = CHEST_XRAY_EXAMPLES
        return_format = MEDICAL_KNOWLEDGE_RETURN
    
    failure_ids = []
    for batch in tqdm(missing_dataset, desc="Knowledge Modeling"):
        idx = batch["missing_idx"]
        
        # Skip the succesful part.
        if os.path.exists(os.path.join(config.save_path, f'{idx}.json')):
            continue
        
        logger.debug(batch)
        # Must to clear history for each new batch.
        proto.reset_history()
        input_type = batch['input_type']
        text_content = batch['text_content']
        image_content = batch['image_content']
        GT_text = batch['GT_text']
        GT_image = batch['GT_image']
        try:
            result = proto.action(knowledge=domain_knowledge,
                                    return_format=return_format,
                                    input_type=input_type,
                                    text_content=text_content,
                                    image_content=image_content,
                                    max_token=config.max_token,
                                    temperature=config.temperature,
                                    convert_json=True)
            logger.debug(result)
            result = proto.postproccess(result)
            final_result = {
                "idx": idx,
                "input_format": input_type,
                "text_content": text_content,
                "image_content": image_content,
                "GT_text": GT_text,
                "GT_image": GT_image,
                "LLM_result": result
            }
            
            
            save_json(final_result, os.path.join(config.save_path, f'{idx}.json'))

        except Exception as e:
            failure_ids.append(idx)
            logger.error(e)
            continue


    return failure_ids



def main(config):
    inference_config = config.inference
    dataset_config = config.dataset
    lmm_config = config.lmm
    logger = get_logger()
    
    logger.info(f"Config: {config}")
    
    reproducibility_setting(inference_config.seed)
    logger.info(f"Set global seed as: {inference_config.seed}.")
    os.makedirs(inference_config.save_path, exist_ok=True)
    
    # get dataset merge missing.
    logger.info("Processing the missing dataset....")
    missing_dataset, text_num_missing, image_num_missing = get_missing_dataset(dataset_config)
    logger.info(f"Dataset processed. Missing numbers: text: {text_num_missing}, image: {image_num_missing}")
    
    # Initialize LMM
    logger.info("Initialize LMM agent....")
    agent = LMAgent(lmm_config.base_url, lmm_config.api_key, lmm_config.model_name)
    logger.info("Initialize LMM agent.... [Finish]")
    
    # load knowledge module.
    logger.info("Load knowledge modeling module....")
    if inference_config.domain == 'general':
        KE_proto = KnowledgeExtractingProto(agent, logger=logger)
    elif inference_config.domain == 'medical':
        KE_proto = MedicalKnowledgeExtractionProto(agent, logger=logger)
    else:
        raise ValueError(f'Domain {inference_config.domain} is not defined.')
    logger.info("Load knowledge modeling module.... [Finish]")
    
    # knowledge modeling.
    failure_ids = knowledge_modeling(KE_proto, missing_dataset, logger, inference_config)
    save_json(failure_ids, os.path.join(inference_config.save_path, 'failure_ids.json'))
    logger.info(f"Modeling done! Fails: {len(failure_ids)}. ")
    
    
    
if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    main(config)