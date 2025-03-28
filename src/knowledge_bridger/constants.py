# GENERAL_ELEMENTS = "entitis, scenarios, actions (or interactions), and image style"
from pathlib import Path


INTEGRATED_PROMPT = """
# Instruction
- You have to integrate the previous result into a sturcture format.
- Use precise nouns and avoid general terms; each object should be accurately named.

# Return Format
The output must be in JSON format as follows:

{return_format}
"""


GENERAL_ELEMENTS = """
1. Identify the top 7 objects by their specific names (e.g., `man`, `woman` instead of `person`).
2. Specify the count of each identified object.
3. Describe attributes for each object in detail.
4. Summarize the style of the {input_format}."""


MEDICAL_ELEMENTS_IMAGE2REPORT = """
This is a chest X-ray image. Please follow these steps for a comprehensive analysis:
- Describe the main anatomical structures visible in the image, such as the lungs, heart, and trachea.
- Identify any abnormalities present, such as opacities, nodules, or effusions, and describe their characteristics.
- Explain the potential clinical significance of any abnormalities noted.
- Summarize the findings and draft a detailed clinical report based on your observations."""


MEDICAL_ELEMENTS_REPORT2IMAGE = """
Given the following clinical report, analyze and identify specific visual details that would correspond to the described findings on a chest X-ray. Follow these steps:
- Identify the main anatomical structures mentioned in the report and locate them on a chest X-ray.
- Highlight the abnormalities or specific findings described in the report.
- Describe the characteristics (e.g., size, shape, density) of these abnormalities.
- Relate these characteristics to potential clinical conditions.
- Summarize your analysis with a list of visual features expected in the X-ray.

"""

MEDICAL_KNOWLEDGE_RETURN = """
### Structured Analysis
1. **Anatomical Structures**:
   - Lungs: [Left Upper Lobe: Normal/Abnormal], [Right Lower Lobe: Normal/Abnormal]
   - Heart: [Normal/Abnormal]
   - Trachea: [Normal/Abnormal]

2. **Type of Abnormality**:
   - Identified Abnormality: [e.g., opacity, nodule, effusion]
   - Characteristics: [e.g., size: 2 cm, shape: round, border: well-defined/ill-defined, density: high]

3. **Distribution and Location**:
   - Side: [Unilateral/Bilateral]
   - Location: [Upper/Lower/Middle lobe]
   - Extent: [Localized/Diffuse]

4. **Clinical Implication**:
   - Possible Diagnosis: ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices']
   - Recommended Action: [Further imaging, clinical follow-up, etc.]


### Return Format
Please analyze the given information and return your analysis in the following JSON format:

```
"""

"""
Return example:
```
{
  "anatomical_structures": {
    "lungs": {
      "left_upper_lobe": "Normal",
      "right_lower_lobe": "Abnormal"
    },
    "heart": "Normal",
    "trachea": "Normal"
  },
  "abnormality_details": {
    "type": "opacity",
    "characteristics": {
      "size": "approximately 3 cm",
      "shape": "irregular",
      "border": "ill-defined",
      "density": "high"
    }
  },
  "distribution_and_location": {
    "side": "Unilateral",
    "location": "right lower lobe",
    "extent": "Localized"
  },
  "clinical_implication": {
    "possible_diagnosis": "pneumonia",
    "recommended_action": "further clinical evaluation and comparison with previous imaging"
  },
}

Your response:
"""

MEDICAL_GENERATION_INPUT_CONTENT = """
According to the provided information, refine the following basic report: 
{report}
"""

MEDICAL_GENERATION_RETURN = """
Using the following structured analysis, this information is organized to generate {num_prompts} meaningful clinical reports:

### Structured Analysis
{structured_analysis}

{input_content}

### Output Format (in JSON)
[
  "report 1",
  "report 2",
  ...,
  "report {num_prompts}"
]
"""


CHEST_XRAY_EXAMPLES = [
    {
        "report": '1. Increased opacity in the right upper lobe with XXXX associated atelectasis may represent focal consolidation or mass lesion with atelectasis. Recommend chest CT for further evaluation. 2. XXXX opacity overlying the left 5th rib may represent focal airspace disease.\nThere is XXXX increased opacity within the right upper lobe with possible mass and associated area of atelectasis or focal consolidation. The cardiac silhouette is within normal limits. XXXX opacity in the left midlung overlying the posterior left 5th rib may represent focal airspace disease. No pleural effusion or pneumothorax. No acute bone abnormality.',
        "file_name": '[your path]/IU-XRay/images/CXR1000_IM-0003-1001.png'
    },
    
    {
        "report": 'Diffuse fibrosis. No visible focal acute disease.\nInterstitial markings are diffusely prominent throughout both lungs. Heart size is normal. Pulmonary XXXX normal.',
        "file_name": '[your path]/IU-XRay/images/CXR1001_IM-0004-1001.png'
    }
]


GENERAL_ELEMENTS_RETURN = """
{{
    "objects": ["Obj. 1", "Obj. 2", ...],
    "numbers": {{
       "Obj. 1": 2,
       "Obj. 2": 1,
       ...
    }},
    "attributes": {{
       "Obj. 1": "Description of attributes here.",
       ...
    }},
    "style": "Description of style here."
}}
"""

KNOWLEDGE_GRAPH_RETURN = """
[
    {{
        "head": ...,
        "relation": ...,
        "tail": ...
    }},
    ...
]
"""

GENERATE_TEXT_RETURN = """[
    "caption 1",
    ....,
    "caption K"
]"""



# LM setting. According to your deploy setting, modify these constants.
VLM_BASE_URL = 'http://localhost:12345/v1'

API_KEY = 'token-abc123'

PREFIX = ''

VLM_MODEL_NAME = f'{PREFIX}microsoft/Phi-3.5-vision-instruct'
LLAVA_MODEL_NAME = f'{PREFIX}llava-hf/llava-v1.6-vicuna-7b-hf'
QWEN2VL_2B_MODEL_NAME = f'{PREFIX}Qwen/Qwen2-VL-2B-Instruct'
QWEN2VL_7B_MODEL_NAME = f'{PREFIX}Qwen/Qwen2-VL-7B-Instruct'
QWEN2VL_72B_MODEL_NAME = f'{PREFIX}Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8'
# ------------------------------------------------------------------------


# if __name__ == '__main__':
#     print(LLM_MODEL_NAME)