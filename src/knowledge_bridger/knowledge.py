from typing import Dict, List
import os

from tqdm import tqdm
import torch

from .base import KBBase
from .large_model_agents import LMAgent
from .constants import KNOWLEDGE_GRAPH_RETURN, INTEGRATED_PROMPT
from .constants import (MEDICAL_ELEMENTS_REPORT2IMAGE, 
                        MEDICAL_ELEMENTS_IMAGE2REPORT)
from .constants import MEDICAL_GENERATION_RETURN, MEDICAL_GENERATION_INPUT_CONTENT



class KnowledgeExtractingProto(KBBase):

    def __init__(self, llm_agent: LMAgent, logger=None):
        super().__init__()
        self.agent = llm_agent
        self.logger = logger

    def action_batch(self,
                     batches,
                     knowledge,
                     return_format,
                     save_times=16,
                     ckpt_path=None,
                     max_token=256,
                     temperature=0.,
                     seed=42,
                     resume=True,
                     ) -> List[dict]:
        """
        Args:
            ....
        Return:
            A dict contains raw input and (characterizes and knowledge_graph).
        """
        if resume and ckpt_path is not None:
            torch.load(os.path.join(ckpt_path, 'ckpt.pt'))
        results = []
        processed_idx = []
        failure_ids = []
        for idx, batch in enumerate(tqdm(batches)):
            if idx in processed_idx:
                continue
            # Must to clear history for each new batch.
            self.reset_history()
            input_type = batch['input_type']
            text_content = batch['text_content']
            image_content = batch['image_content']
            GT_text = batch['GT_text']
            GT_image = batch['GT_image']
            try:
                result = self.action(knowledge=knowledge,
                                     return_format=return_format,
                                     input_type=input_type,
                                     text_content=text_content,
                                     image_content=image_content,
                                     max_token=max_token,
                                     temperature=temperature,
                                     convert_json=True)
                result = self.postproccess(result)
                results.append({
                    "idx": idx,
                    "input_format": input_type,
                    "text_content": text_content,
                    "image_content": image_content,
                    "GT_text": GT_text,
                    "GT_image": GT_image,
                    "LLM_result": result
                })

                processed_idx.append(idx)
            except Exception as e:
                failure_ids.append(idx)
                if self.logger is not None:
                    self.logger.error(e)
                else:
                    print(e)

            #
            if (idx + 1) % save_times and ckpt_path is not None == 0:
                torch.save((results, processed_idx),
                           os.path.join(ckpt_path, 'ckpt.pt'))

        # final.
        if ckpt_path is not None:
            torch.save((results, processed_idx, failure_ids),
                       os.path.join(ckpt_path, 'final.pt'))

        return results, failure_ids

    def action(self,
               knowledge,
               return_format,
               input_type='text',
               text_content='',
               image_content=None,
               max_token=256,
               temperature=0.,
               convert_json=True):
        result = {}
        # Step 1: extracting the characterizes from the given modality.
        cot_prompt, integrated_prompt = self.characterizes_prompt(
            knowledge, return_format, input_type)

        if input_type == 'text':
            cot_prompt = cot_prompt.format(
                content=text_content, input_format=input_type)
        else:
            cot_prompt = cot_prompt.format(content="", input_format=input_type)
        self.append_msg('user', text_content=cot_prompt,
                        image_content=image_content)
        cot_result = self.agent.chat(
            self.msg_history, max_token=max_token, temperature=temperature)
        self.append_msg('assistant', cot_result)
        # integrated
        self.append_msg('user', text_content=integrated_prompt,
                        image_content=None)
        charac = self.agent.chat(
            self.msg_history, max_token=max_token, temperature=temperature)
        self.append_msg('assistant', charac)
        try:
            if convert_json:
                charac = self.convert_result_to_json(charac)
            result['characterizes'] = charac
        except Exception as e:
            raise RuntimeError(f'Convert json failed at step 1. {charac}')

        # Step 2: Based on characterizes, extracting the enetities and relationships to build the knowledge graph.
        cot_prompt, integrated_prompt = self.knowledge_graph_prompt(
            knowledge, KNOWLEDGE_GRAPH_RETURN, input_type=input_type)
        if input_type == 'text':
            cot_prompt = cot_prompt.format(content=text_content)
        else:
            cot_prompt = cot_prompt.format(content="")
        self.append_msg('user', text_content=cot_prompt,
                        image_content=image_content)
        cot_result = self.agent.chat(
            self.msg_history, max_token=max_token, temperature=temperature)
        self.append_msg('assistant', cot_result)

        # integrated
        self.append_msg('user', text_content=integrated_prompt,
                        image_content=None)
        kg = self.agent.chat(
            self.msg_history, max_token=max_token, temperature=temperature)
        self.append_msg('assistant', kg)
        try:
            if convert_json:
                kg = self.convert_result_to_json(kg)
            result['knowledge_graph'] = kg
        except:
            raise RuntimeError(f'Convert json failed at step 2. {kg}')
        return result

    def characterizes_prompt(self, knowledge, return_format, input_type='image'):
        assert input_type in ['image', 'text'], f"Input format must be `image` or `text`, but got {input_type}."

        cot_prompt = f"""
# Instruction
Understand the given {input_type} to answer the following points with no more than 7 objects:
{knowledge}

{'- User input:' if input_type == 'text' else ''} {{content}}
Please process each point step by step.
"""

        integrated_prompt = INTEGRATED_PROMPT.format(
            return_format=return_format)

        return cot_prompt, integrated_prompt

    def knowledge_graph_prompt(self, knowledge, return_format=None, input_type='image'):
        assert input_type in [
            'image', 'text'], f"Input format must be `image` or `text`, but got {input_type}."

        cot_prompt = f"""
# Instruction
Your task is to analyze the provided {input_type} and extract **exactly 10 distinct relationships** to build a knowledge graph. Each relationship should be structured as (Head, Relation, Tail), focusing on **clear, direct relations** (e.g., "causes," "is a part of," "describes," etc.). 

{"Input text:" if input_type == 'text' else ""} {{content}}

Think through the relationships step-by-step.        
"""

        integrated_prompt = INTEGRATED_PROMPT.format(
            return_format=return_format)

        return cot_prompt, integrated_prompt

    def system_role(self):
        return super().system_role()

    def postproccess(self, result):
        """
        Drop repeat and illegal content.
        """
        result['characterizes']['objects'] = list(set(result['characterizes']['objects']))
        # result['characterizes']['attributes'] = list(set(result['characterizes']['attributes']))
        knowledge_graph = []
        for item in result['knowledge_graph']:
            if 'head' not in item:
                continue
            if 'relation' not in item:
                continue
            if 'tail' not in item:
                continue
            knowledge_graph.append(item)
        result['knowledge_graph'] = knowledge_graph
        return result


class KnowledgeDrivenGenerationProto(KBBase):

    def __init__(self, llm_agent: LMAgent, logger=None):
        super().__init__()
        self.agent = llm_agent
        self.logger = logger

    def action(self,
               knowledge,
               return_format,
               input_type='text',
               text_content='',
               image_content=None,
               max_token=256,
               temperature=0.,
               num_captions=10,
               convert_json=True) -> Dict:
        if input_type == 'text':
            content = text_content
        else:
            content = ''
        cot_prompt, integrated_prompt = self.consturct_prompt(knowledge,
                                                              return_format,
                                                              input_type,
                                                              content,
                                                              num_captions=num_captions)

        # step 1, cot.

        self.append_msg('user', text_content=cot_prompt,
                        image_content=image_content)
        cot_result = self.agent.chat(
            self.msg_history, max_token=max_token, temperature=temperature)
        self.append_msg('assistant', cot_result)

        if input_type == 'image':
            # Step 2.1: generate captions.
            self.append_msg(
                'user', text_content=integrated_prompt, image_content=None)
            result = self.agent.chat(
                self.msg_history, max_token=max_token, temperature=temperature)
        elif input_type == 'text':
            import re
            results = []
            # Step 2.2: generate image prompt.
            # For eluminating hallucination, we iteratively generate result with multi-turn style.
            df_prompt, iterative_prompt = self.consturct_diffusion_prompt(knowledge,
                                                                          text_content=text_content,
                                                                          num_prompts=num_captions)
            refining_prompt = self.refining_prompt()
            self.append_msg('user', text_content=df_prompt, image_content=None)
            tmp = self.agent.chat(
                self.msg_history, max_token=max_token, temperature=temperature)
            self.append_msg('assistant', text_content=tmp, image_content=None)
            results.append(re.findall(r'"(.*)"', tmp)[0])
            for idx in range(1, num_captions):
                self.append_msg('user',
                                text_content=iterative_prompt.format(previous=tmp,
                                                                     generated_idx=idx,
                                                                     next_idx=idx+1),
                                image_content=None)
                tmp = self.agent.chat(
                    self.msg_history, max_token=max_token, temperature=temperature)
                results.append(re.findall(r'"(.*)"', tmp)[0])
                self.append_msg('assistant', text_content=tmp,
                                image_content=None)
            new_prompts = []    
            for description in results:
                self.reset_history()
                self.append_msg('user', refining_prompt.format(description=description))
                prompt = self.agent.chat(self.msg_history, max_token=max_token, temperature=temperature)
                try:
                    prompt = re.findall(r'"(.*)"', prompt)[0]
                except:
                    prompt = prompt.replace('"', '')
                new_prompts.append(prompt)
                
            return cot_result, new_prompts

        else:
            raise ValueError(
                f"Input format must be `image` or `text`, but got {input_type}.")

        # Step 3: convert final result.
        self.append_msg('assistant', result)
        try:
            if convert_json:
                result = self.convert_result_to_json(result)
        except:
            raise RuntimeError(f'Convert json failed. {result}')

        result = self.postproccess(result)

        return cot_result, result

    def action_batch(self, *args, **kwargs) -> List[Dict]:
        return super().action_batch(*args, **kwargs)

    def consturct_diffusion_prompt(self, knowledge, text_content, num_prompts=10):
        prompt = f"""

# Instruction
- Expand the basic sentence to {num_prompts} high-quality description based on previous analysis and structured data.
- Each new prompt should emphasize different object attributes or scene details.

- **Basic Sentence**: {text_content}

*Structured Data*
- **Objects**: {knowledge['characterizes']['objects']}
- **Numbers**: {knowledge['characterizes']['numbers']}
- **Attributes**: {knowledge['characterizes']['attributes']}
- **Relationships**: {knowledge['knowledge_graph']}
- **Style**: {knowledge['characterizes']['style']}

Ok, let's generate the first prompt.

# Output Format
Output the prompt as a single:
prompt 1: "Generated Prompt"
"""
        iterative_prompt = """
# Instruction
1. You have currently generated {generated_idx} prompt(s).
2. Based on the previous result, generate the {next_idx}-th prompt. Ensure this prompt is distinct, focusing on new visual or descriptive details, while maintaining continuity with the previous prompts. Avoid repeating previous phrases or descriptions.

# Tips
- Highlight different attributes, perspectives, or compositional aspects for each new prompt.
- Use varied descriptive language to expand upon visual style, lighting, atmosphere, or object relationships.

# Previous prompt
{previous}

# Output Format
Output the {next_idx}-th prompt as a single:
"prompt {next_idx}": "Generated Prompt"

"""
        return prompt, iterative_prompt
    
    
    def refining_prompt(self):
        prompt = """
### Optimization Prompt for LLM

**Objective:** Transform the following detailed description into a single, concise sentence that retains the key elements necessary for effective understanding and implementation within a Stable Diffusion model. 

**Detailed Description:** {description}

**Guidelines:**
1. **Clarity:** Ensure the sentence is clear and free of unnecessary jargon.
2. **Essentials:** Focus on the core elements that are critical for the model.
3. **Conciseness:** Limit the sentence to a maximum of 20-70 words.
4. **Contextual Relevance:** Use terminology and concepts that are compatible with Stable Diffusion principles.

**Example Input:**
"A multi-layered neural network equipped with backpropagation for error correction, designed to enhance the transmission of signals by minimizing noise interference."

**Example Output:**
"A neural network with error-correcting backpropagation for clear signal transmission."

Perform the transformation on the provided description and present the optimized sentence below:
"""
        return prompt

    def consturct_prompt(self, knowledge, return_format, input_type, content, num_captions=10):
        assert input_type in [
            'image', 'text'], f"Input format must be `image` or `text`, but got {input_type}."

        cot_prompt = f"""
# Instruction
Using the provided {input_type} and structured data, generate a detailed, cohesive description of the scene. Focus on synthesizing visual elements, objects, and their relationships to create a natural, flowing narrative.

## Step-by-Step Guide
1. **Object Analysis**: Start by identifying the primary objects in the image, referencing the object names, counts, and attributes provided.
2. **Attribute Integration**: Describe each object’s specific attributes in detail, noting how they visually appear and interact with other elements or the environment.
3. **Relationship Context**: Integrate any relationships (e.g., spatial, ownership, or functional) to add context to the scene. Use logical transitions (e.g., “next to,” “owned by”) to clarify how objects relate to each other.
4. **Style and Atmosphere**: Based on the overall style and aesthetic notes, enrich the description to reflect the mood or setting (e.g., light quality, color tones, or ambiance).

*Provided Data*
- **Objects**: {knowledge['characterizes']['objects']}
- **Numbers**: {knowledge['characterizes']['numbers']}
- **Attributes**: {knowledge['characterizes']['attributes']}
- **Relationships**: {knowledge['knowledge_graph']}
- **Style**: {knowledge['characterizes']['style']}

{'- User input:' if input_type == 'text' else ''} {content}

# Output Format
Write the description as a cohesive paragraph that seamlessly incorporates all elements, providing a vivid and natural interpretation of the scene.
"""

        integrated_prompt = f"""
# Instruction
1. Carefully read and understand the previous paragraph to capture its main ideas, tone, and style.
2. Generate {num_captions} unique, meaningful, and contextually diverse captions that provide a reasonable summary of the content. Aim for a range of perspectives, such as highlighting different visual details, emotions, or themes.

# Return Format
The output should be in JSON format as follows:

{return_format}
"""
        return cot_prompt, integrated_prompt

    def postproccess(self, result):
        result = list(set(result))
        return result



class MedicalKnowledgeExtractionProto(KBBase):
    
    def __init__(self, llm_agent: LMAgent, logger=None):
        super().__init__()
        self.agent = llm_agent
        self.logger = logger
    
    
    def system_role(self):
        return "You are a very experienced radiologist."   
    
    def action(self, knowledge,
               return_format,
               input_type='text',
               text_content='',
               image_content=None,
               max_token=256,
               temperature=0.,
               convert_json=True) -> Dict:
        result = {}
        # Step 1: extracting the characterizes from the given modality.
        fewshot_promt, cot_prompt = self.characterizes_prompt(knowledge, return_format, input_type)
        # Using fewshot learning.
        self.append_msg('user', text_content=fewshot_promt, image_content=[knowledge[0]['file_name'], knowledge[1]['file_name']])
        fewshot_result = self.agent.chat(self.msg_history, max_token=max_token, temperature=temperature)
        self.append_msg('assistant', text_content=fewshot_result, image_content=None)
        
        # Step 2: using cot
        if input_type == 'text':
            cot_prompt = cot_prompt.format(input_content=text_content, return_format=return_format)
        else:
            cot_prompt = cot_prompt.format(input_content="", return_format=return_format)
        self.append_msg('user', text_content=cot_prompt,
                        image_content=image_content)
        cot_result = self.agent.chat(self.msg_history, max_token=max_token, temperature=temperature)
        
        try:
            if convert_json:
                cot_result = self.convert_result_to_json(cot_result)
            result = cot_result
        except:
            raise RuntimeError(f'Convert json failed at step 2. {cot_result}')
        return result
    
    
    
    def action_batch(self, *args, **kwargs) -> List[Dict]:
        return super().action_batch(*args, **kwargs)
    
    
    def characterizes_prompt(self, knowledge, return_format, input_type='image'):
        assert input_type in ['image', 'text'], f"Input format must be `image` or `text`, but got {input_type}."

        fewshot_prompt = f"""
I will give you some chest x-ray image and report examples. You need understand the images and reports.
# Examples
Example 1:
Chest X-ray Image: See Image 1.
Clinical report: {knowledge[0]['report']}.

Example 2:
Chest X-ray Image: See Image 2.
Clinical report: {knowledge[1]['report']}.
"""     
        cot_prompt = f"""
{MEDICAL_ELEMENTS_IMAGE2REPORT if input_type == 'image' else MEDICAL_ELEMENTS_REPORT2IMAGE}

{{input_content}}

{{return_format}}
"""
        return fewshot_prompt, cot_prompt
    
    
    def postproccess(self, result):
        return result
        
    
    
class MedicalKnowledgeGenerationProto(KBBase):
    
    def __init__(self, llm_agent: LMAgent, logger=None):
        super().__init__()
        self.agent = llm_agent
        self.logger = logger
        
    def action(self, knowledge,
               return_format,
               input_type='text',
               text_content='',
               image_content=None,
               max_token=256,
               temperature=0.,
               num_captions=10,
               convert_json=True) -> Dict:
        if input_type == 'text':
            prompt = self.consturct_prompt(knowledge, text_content, input_type=input_type, num_prompts=num_captions)
        else:
            prompt = self.consturct_prompt(knowledge, image_content, input_type=input_type, num_prompts=num_captions)
            
        self.append_msg('user', text_content=prompt, image_content=image_content)
        result = self.agent.chat(self.msg_history, max_token=max_token, temperature=temperature)
        try:
            if convert_json:
                result = self.convert_result_to_json(result)
        except:
            raise RuntimeError(f'Convert json failed at. {result}')
        return None, result
        

    
    def consturct_prompt(self, knowledge, input_content, input_type='text', num_prompts=5):
        assert input_type in ['image', 'text'], f"Input format must be `image` or `text`, but got {input_type}."
        prompt = MEDICAL_GENERATION_RETURN.format(num_prompts=num_prompts, 
                                                  structured_analysis=knowledge,
                                                  input_content=input_content)
        return prompt