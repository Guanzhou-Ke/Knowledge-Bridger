import argparse
import os
import shutil
import numpy as np
import torch
import networkx as nx

from utils import (load_config,
                   reproducibility_setting,
                   load_json_from_file,
                   save_json)
from misc.mylogging import get_logger
from metrics.CLIPScore import CLIPScore
from metrics.BLIPScore import BLIPScore
from metrics.graph_similarity import jaccard_similarity, graph_cosine_similarity
from knowledge_bridger.large_model_agents import LMAgent
from knowledge_bridger import (GENERAL_ELEMENTS, 
                               GENERAL_ELEMENTS_RETURN, 
                               KnowledgeExtractingProto)
from knowledge_bridger import  (MedicalKnowledgeExtractionProto,
                                CHEST_XRAY_EXAMPLES, 
                                MEDICAL_KNOWLEDGE_RETURN)


def load_ranking_model(config, device):
    clip_score = CLIPScore(download_root=config.clip_model_path, device=device,)
    blip_score = BLIPScore(model_name=config.blip_model_path, device=device)
    if config.graph_similarity_method == 'cosine':
        graph_similarity = graph_cosine_similarity
    elif config.graph_similarity_method == 'jaccard':
        graph_similarity = jaccard_similarity
    else:
        raise ValueError(
            f"Graph similarity method {config.graph_similarity_method} is not supported.")
    return clip_score, blip_score, graph_similarity


def parse_args():
    parser = argparse.ArgumentParser(prog='Knowledge extraction.',)

    parser.add_argument('--config', '-f', help='Config file path')

    args = parser.parse_args()
    return args


def build_graph(knowledge_graphs):
    """
    Use networkx to build a graph from the knowledge graph dictionary.
    The dict is in the format of {head: xxx, relation: xxx, tail: xxx}.
    """
    graph = nx.Graph()
    for kg in knowledge_graphs:
        head = kg['head']
        relation = kg['relation']
        tail = kg['tail']
        graph.add_node(head)
        graph.add_node(tail)
        graph.add_edge(head, tail, relation=relation)
    return graph


def compute_ss(clip_score, blip_score, path, num_gene=7):
    knowledge = load_json_from_file(os.path.join(path, 'knowledge.json'))
    if os.path.exists(os.path.join(path, 'prompts.json')):
        prompts = load_json_from_file(os.path.join(path, 'prompts.json'))
    else:
        prompts = load_json_from_file(os.path.join(path, 'texts.json'))
    avaliable_modality = knowledge['input_format']
    if avaliable_modality == 'text':
        GT = knowledge['GT_text']
        generated_images = [os.path.join(path, f'id_{i}.jpg') for i in range(num_gene)]
        input_texts = [GT]
        input_images = generated_images
    else:
        GT = knowledge['GT_image']
        input_texts = prompts
        input_images = [GT]
        
    c_score = clip_score.score(input_texts, input_images)
    b_score = blip_score.score(input_texts, input_images)
    scores = (np.array(c_score) + np.array(b_score)) / 2
    return scores


def knowledge_modeling(proto, candidates, logger, input_type='text'):
    
    if config.domain == 'general':
        domain_knowledge = GENERAL_ELEMENTS
        return_format = GENERAL_ELEMENTS_RETURN
    elif config.domain == 'medical':
        domain_knowledge = CHEST_XRAY_EXAMPLES
        return_format = MEDICAL_KNOWLEDGE_RETURN
    failure_ids = []
    for idx, sample in enumerate(candidates):
        logger.debug(sample)
        # Must to clear history for each new batch.
        proto.reset_history()
        text_content = sample if input_type == 'text' else None
        image_content = sample if input_type == 'image' else None
        try:
            result = proto.action(knowledge=domain_knowledge,
                                    return_format=return_format,
                                    input_type=input_type,
                                    text_content=text_content,
                                    image_content=image_content,
                                    max_token=512,
                                    temperature=0.3,
                                    convert_json=True)
            logger.debug(result)
            result = proto.postproccess(result)
            final_result = {
                "input_format": input_type,
                "text_content": text_content,
                "image_content": image_content,
                "LLM_result": result
            }
            
            save_json(final_result, os.path.join(config.save_path, f'candi_{idx}_kg.json'))

        except Exception as e:
            failure_ids.append(idx)
            logger.error(e)
            continue


    return failure_ids

def compute_graph_similarity(graph_func, path, ke_proto, logger, num_gene=7):
    if os.path.exists(os.path.join(path, 'prompts.json')):
        input_type = 'image'
    else:
        input_type = 'text'
    ava_modality_knowledge = os.path.join(path, 'knowledge.json')
    ava_graph = build_graph(ava_modality_knowledge['LLM_result']['knowledge_graph'])
    
    # generate the knowledge graph for each candidate.
    if input_type == 'text':
        candidates = load_json_from_file(os.path.join(path, 'texts.json'))
    else:
        # search the image files, .jpg.
        candidates = [os.path.join(path, f'id_{i}.jpg') for i in range(num_gene)]
    knowledge_modeling(ke_proto, candidates, logger, input_type=input_type)
    scores = []
    # load each candidate's knowledge graph.
    for idx in range(num_gene):
        candidate_path = os.path.join(path, f'candi_{idx}_kg.json')
        if not os.path.exists(candidate_path):
            raise ValueError(f"Candidate {idx} knowledge graph file not found in {path}.")
        candidate_knowledge = load_json_from_file(candidate_path)
        candidate_graph = build_graph(candidate_knowledge['LLM_result']['knowledge_graph'])
        
        # compute the graph similarity.
        graph_similarity = graph_func(ava_graph, candidate_graph)
        scores.append(graph_similarity) 
        
    return np.array(scores)
    
def ranking(clip_score, blip_score, graph_func, config, ke_proto, logger):
    """
    Rank the generated knowledge graph based on the similarity score.
    """
    sub_dirs = os.listdir(config.generation_dir)
    for sub in sub_dirs:
        # check if the sub_dir is a directory
        if not os.path.isdir(os.path.join(config.generation_dir, sub)):
            continue
        # compute semantic score
        semantic_score = compute_ss(clip_score, blip_score, os.path.join(config.generation_dir, sub), num_gene=config.num_generations)
        # extract the generated modality's knowledge graph.
        graph_score = compute_graph_similarity(graph_func, os.path.join(config.generation_dir, sub), ke_proto, logger, num_gene=config.num_generations)
        total_score = semantic_score + graph_score
        # select the best candidate.
        best_candidate = np.argmax(total_score)
        results = {
            "semantic_score": semantic_score,
            "graph_score": graph_score,
            "total_score": total_score,
            "best_candidate": best_candidate
        }
        save_json(results, os.path.join(config.generation_dir, sub, 'ranking.json'))


def main(config):
    inference_config = config.inference
    ranking_config = config.ranking
    lmm_config = config.lmm
    logger = get_logger()

    device = torch.device(f'cuda:{inference_config.gpu_ids}')

    logger.info(f"Config: {config}")

    reproducibility_setting(inference_config.seed)
    logger.info(f"Set global seed as: {inference_config.seed}.")
    
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
    
    clip_score, blip_score, graph_func = load_ranking_model(ranking_config, device)
    
    ranking(clip_score, blip_score, graph_func, ranking_config, KE_proto, logger)
    


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    main(config)
