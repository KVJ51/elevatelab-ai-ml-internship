from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class MentalHealthChatbot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

    def get_response(self, user_input):
        inputs = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        return response
