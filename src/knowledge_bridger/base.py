from typing import List, Dict
import json_repair
import base64


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


class KBBase:

    def __init__(self):
        self.msg_history = []
        self.init_system_role()
        pass

    def system_role(self):
        return "You are a helpful assistant in understanding images and texts and you can extract very important and accurate information from them."

    def wrap_result(self, result):
        return self.convert_result_to_json(result)
    
    def action(self, *args, **kwargs) -> Dict:
        """
        Perform a pre-defined task or action.
        Return a dict.
        """
        raise NotImplementedError


    def construct_prompt(self, *args, **kwargs) -> str:
        """
        Construct prompt.
        """
        raise NotImplementedError

    def action_batch(self, *args, **kwargs) -> List[Dict]:
        """
        Perform a pre-defined task or action within batch.
        Return a list including dict.
        """
        raise NotImplementedError

    
    def memory(self, msg):
        self.msg_history.append(msg)

    def save_result(self, format='json'):
        # TODO: save result.
        """
        Save agent's results. 

        """
        if format == 'json':
            pass

    def init_system_role(self):
        self.msg_history.append({
            "role": "system",
            "content": [
                {"type": "text", "text": self.system_role()},
            ],
        })

    def reset_history(self):
        self.msg_history = []
        self.init_system_role()

    def append_msg(self, role, text_content, image_content=None):
        msg = {
            "role": role,
            "content": [
                {"type": "text", "text": text_content},
            ]
        }

        if image_content is not None:
            for path in image_content:
                base64_image = encode_image(path)
                msg['content'].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                )

        self.msg_history.append(msg)

    def convert_result_to_json(self, result: str):
        decoded_object = json_repair.loads(result)
        return decoded_object
    
    

    
