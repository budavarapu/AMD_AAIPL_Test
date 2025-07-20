#
# Phi-4-mini-Instruct in action.
import re
import time
import torch
from typing import Optional, Union, List
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.random.manual_seed(0)
class AAgent(object):
    def __init__(self, **kwargs):
        # self.model_type = input("Available models are Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct. Enter 1B or 3B: ").strip()
        self.model_type = kwargs.get('model_type', '3B').strip()
        # model_name = "meta-llama/Llama-3.2-3B-Instruct"
        model_name = "/home/user/hf_models/Llama-3.2-3B-Instruct"
        
        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set left padding for decoder-only models (fixes the warning)
        self.tokenizer.padding_side = 'left'

    def generate_response(self, message: Union[str, List[str]], system_prompt: Optional[str] = None, **kwargs) -> Union[str, List[str]]:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant. Answer the user's question to the best of your ability."
        
        # Convert single message to list for uniform processing
        if isinstance(message, str):
            message = [message]
            single_input = True
        else:
            single_input = False

        # Prepare all messages for batch processing
        all_messages = []
        for msg in message:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg}
            ]
            all_messages.append(messages)
        
        # Convert all messages to text format
        texts = []
        for messages in all_messages:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)

        # Tokenize all texts together with padding
        model_inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.model.device)

        # Conduct batch text completion
        tgps_show_var = kwargs.get('tgps_show', False)
        if tgps_show_var: start_time = time.time()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=kwargs.get('max_new_tokens', 1024),
                temperature=kwargs.get('temperature', 0.1),
                top_p=kwargs.get('top_p', 0.9),
                do_sample=kwargs.get('do_sample', True),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        if tgps_show_var:
            gen_time = time.time() - start_time
            token_len = 0

        # Decode the batch
        batch_outs = []
        for i, (input_ids, generated_sequence) in enumerate(zip(model_inputs.input_ids, generated_ids)):
            # Extract only the newly generated tokens
            output_ids = generated_sequence[len(input_ids):].tolist()
            if tgps_show_var: token_len += len(output_ids)
            # Decode the full result
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            # Clean up unwanted "assistant" prefix using regex
            content = re.sub(r'^assistant\s*\n*', '', content, flags=re.IGNORECASE).strip()
            
            batch_outs.append(content)
        if tgps_show_var:
            return batch_outs[0] if single_input else batch_outs, token_len, gen_time
        # Return single string if input was single string, otherwise return list
        return batch_outs[0] if single_input else batch_outs, None, None
        
if __name__ == "__main__":
    # Single message (backward compatible)
    ans_agent = AAgent()
    response, tl, gt = ans_agent.generate_response("Solve: 2x + 5 = 15", system_prompt="You are a math tutor.", tgps_show=True, max_new_tokens=512, temperature=0.1, top_p=0.9, do_sample=True)
    print(f"Single response: {response}")
    print(f"Token length: {tl}, Generation time: {gt:.2f} seconds, Tokens per second: {tl/gt:.2f}")
    print("-----------------------------------------------------------")
          
    # Batch processing (new capability)
    messages = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "What are the main differences between Python and Java?",
        "What is the significance of the Turing Test in AI?",
        "What is the capital of Japan?",
    ]
    responses, tl, gt = ans_agent.generate_response(messages, max_new_tokens=512, temperature=0.1, top_p=0.9, do_sample=True, tgps_show=True)
    print("Responses:")
    for i, resp in enumerate(responses):
        print(f"Message {i+1}: {resp}")
    print(f"Token length: {tl}, Generation time: {gt:.2f} seconds, Tokens per second: {tl/gt:.2f}")
    print("-----------------------------------------------------------")

    # Custom parameters
    response = ans_agent.generate_response(
        "Write a story", 
        temperature=0.8, 
        max_new_tokens=512
    )
    print(f"Custom response: {response}")
