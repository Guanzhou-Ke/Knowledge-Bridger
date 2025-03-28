import argparse
import os
import shutil

from tqdm import tqdm
from diffusers import DiffusionPipeline
import torch
from torchvision.transforms.functional import to_pil_image

from cheff import CheffLDMT2I
from utils import (load_config,
                   reproducibility_setting,
                   load_json_from_file,
                   save_json)
from misc.mylogging import get_logger
from knowledge_bridger.large_model_agents import LMAgent
from knowledge_bridger import KnowledgeDrivenGenerationProto, MedicalKnowledgeGenerationProto
from knowledge_bridger.constants import GENERATE_TEXT_RETURN


def load_image_generator(config, device):
    if config.domain == 'general':
        # load both base & refiner
        base = DiffusionPipeline.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        base.to(device)
        if config.use_refiner:
            refiner = DiffusionPipeline.from_pretrained(
                config.refiner_model_name,
                text_encoder_2=base.text_encoder_2,
                vae=base.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            refiner.to(device)
        else:
            refiner = None
    elif config.domain == 'medical':
        base = CheffLDMT2I(model_path=config.base_model_name,
                           ae_path=config.refiner_model_name, device=device)
        refiner = None
    else:
        raise ValueError(f"Domain {config.domain} is not defined.")
    return base, refiner


def complete_missing_image(generators, prompts, save_path, config, seed=2025):
    base, refiner = generators
    n_steps = config.n_steps
    high_noise_frac = config.high_noise_frac
    generator = torch.manual_seed(seed)
    for idx, prompt in enumerate(prompts):
        if config.use_refiner and config.domain == 'general':
            image = base(
                prompt=prompt,
                num_inference_steps=n_steps,
                denoising_end=high_noise_frac,
                generator=generator,
                output_type="latent",
            ).images
            image = refiner(
                prompt=prompt,
                num_inference_steps=n_steps,
                denoising_start=high_noise_frac,
                generator=generator,
                image=image,
            ).images[0]
        else:
            if config.domain == 'general':
                image = base(
                    prompt=prompt,
                    num_inference_steps=n_steps,
                    denoising_end=high_noise_frac,
                    generator=generator,
                ).images[0]
            elif config.domain == 'medical':
                image = base.sample(
                    conditioning=prompt,
                    sampling_steps=n_steps,
                    eta=1.0,
                    decode=True,
                    temperature=config.temperature
                )
                image.clamp_(-1, 1)
                image = (image + 1) / 2
                image = to_pil_image(image[0])

        image.save(os.path.join(save_path, f'id_{idx}.jpg'))


def parse_args():
    parser = argparse.ArgumentParser(prog='Knowledge extraction.',)

    parser.add_argument('--config', '-f', help='Config file path')

    args = parser.parse_args()
    return args


def get_missing_knowledge(config):
    missing_knowledge_path = config.missing_knowledge_path
    knowledge_list = os.listdir(missing_knowledge_path)
    knowledges = []
    for file in knowledge_list:
        if file != 'failure_ids.json':
            tmp = load_json_from_file(
                os.path.join(missing_knowledge_path, file))
            knowledges.append(tmp)
    return knowledges


def complete_missing(agent, inference_config, generators, generator_config, knowledges, logger):
    failure_ids = []
    for knowledge in tqdm(knowledges, desc='Missing Generation..'):
        missing_idx = knowledge['idx']
        input_format = knowledge['input_format']
        os.makedirs(os.path.join(inference_config.save_path,
                    str(missing_idx)), exist_ok=True)
        save_json(knowledge, os.path.join(
            inference_config.save_path, str(missing_idx), 'knowledge.json'))
        agent.reset_history()
        try:
            if input_format == 'text':
                cot_result, prompts = agent.action(knowledge['LLM_result'],
                                                   GENERATE_TEXT_RETURN if inference_config.domain == 'general' else None,
                                                   input_type='text',
                                                   image_content=None,
                                                   text_content=knowledge['text_content'],
                                                   max_token=inference_config.max_token,
                                                   temperature=inference_config.temperature,
                                                   num_captions=inference_config.num_candidates)
                complete_missing_image(generators,
                                       prompts,
                                       os.path.join(
                                           inference_config.save_path, str(missing_idx)),
                                       generator_config,
                                       seed=inference_config.seed)
                shutil.copyfile(knowledge['GT_image'], os.path.join(
                    inference_config.save_path, str(missing_idx), 'GT.jpg'))
                save_json(prompts, os.path.join(
                    inference_config.save_path, str(missing_idx), 'prompts.json'))
            elif input_format == 'image':
                cot_result, texts = agent.action(knowledge['LLM_result'],
                                                 GENERATE_TEXT_RETURN,
                                                 input_type='image',
                                                 image_content=knowledge["image_content"],
                                                 max_token=inference_config.max_token,
                                                 temperature=inference_config.temperature,
                                                 num_captions=inference_config.num_candidates)
                save_json(texts, os.path.join(
                    inference_config.save_path, str(missing_idx), 'texts.json'))
            else:
                raise ValueError(
                    f'The input type {input_format} is not supported.')
            save_json(cot_result, os.path.join(
                inference_config.save_path, str(missing_idx), 'cot_result.json'))

        except Exception as e:
            if missing_idx not in failure_ids:
                failure_ids.append(missing_idx)
            logger.error(e)
            continue
    return failure_ids


def main(config):
    inference_config = config.inference
    knowledge_config = config.dataset
    lmm_config = config.lmm
    generator_config = config.generator
    logger = get_logger()

    device = torch.device(f'cuda:{inference_config.gpu_ids}')

    logger.info(f"Config: {config}")

    reproducibility_setting(inference_config.seed)
    logger.info(f"Set global seed as: {inference_config.seed}.")
    os.makedirs(inference_config.save_path, exist_ok=True)

    # # get dataset merge missing.
    logger.info("Loading knowledge....")
    knowledges = get_missing_knowledge(knowledge_config)
    logger.info(f"Loading knowledge...[Finish]")

    # # Initialize LMM
    logger.info(
        f"Initialize LMM agent and {generator_config.domain} generator")
    agent = LMAgent(lmm_config.base_url, lmm_config.api_key,
                    lmm_config.model_name)
    generators = load_image_generator(generator_config, device)

    logger.info(
        f"Initialize LMM agent and {generator_config.domain} generator.... [Finish]")

    # load knowledge module.
    logger.info("Load knowledge-driven Generation module....")
    if inference_config.domain == 'general':
        KDG_proto = KnowledgeDrivenGenerationProto(agent, logger=logger)
    elif inference_config.domain == 'medical':
        KDG_proto = MedicalKnowledgeGenerationProto(agent, logger=logger)
    logger.info("Load knowledge-driven Generation module.... [Finish]")

    # knowledge-driven generation.
    failure_ids = complete_missing(
        KDG_proto, inference_config, generators, generator_config, knowledges, logger)
    save_json(failure_ids, os.path.join(
        inference_config.save_path, 'failure_ids.json'))
    logger.info(f"Modeling done! Fails: {len(failure_ids)}. ")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    main(config)
