#
import os
import re
import time
import torch
from typing import Optional, Union, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel

torch.random.manual_seed(0)
class AAgent(object):
    def __init__(self, adapter_type, **kwargs):
        # self.model_type = input("Available models: Qwen3-1.7B and Qwen3-4B. Please enter 1.7B or 4B: ").strip()
        self.model_type = kwargs.get('model_type', '4B').strip()
        # model_name = "Qwen/Qwen3-4B"
        model_name = "/jupyter-tutorial/hf_models/Qwen3-4B"
        
        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

        self.adapter_type = adapter_type
        
        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self._setup_model_with_adapter(base_model)        
        self.model = self.model.eval()

    def _setup_model_with_adapter(self, base_model):
        """Setup model with or without adapter based on configuration."""
        if self.adapter_type is None:
            print("No adapter specified - using base model")
            return base_model

        self.adapter_type = self.adapter_type.lower()

        if self.adapter_type == 'sft' or self.adapter_type == 'grpo':
            print("Loading required adapter")
            checkpoint_path = self.find_latest_checkpoint()
            if checkpoint_path is None:
                print(f"No trained checkpoints found for {self.adapter_type}")
                inference_model = base_model
            else:
                print(f"Loading LoRA adapters from: {checkpoint_path}")
                inference_model = PeftModel.from_pretrained(base_model, checkpoint_path)
            return inference_model
        else:
            print(f"Unknown adapter type: {self.adapter_type}. Using base model.")
            return base_model

    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in the output directory."""

        output_dir = "checkpoints"
        output_dir = os.path.join(output_dir, self.adapter_type)
        
        checkpoint_pattern = re.compile(r'checkpoint-(\d+)')
        checkpoints = []
        
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                match = checkpoint_pattern.match(item)
                if match:
                    step_num = int(match.group(1))
                    checkpoints.append((step_num, item_path))
        
        if not checkpoints:
            print(f"No checkpoints found in {output_dir}")
            return None
        
        checkpoints.sort(key=lambda x: x[0])
        latest_checkpoint = checkpoints[-1][1]
        
        print(f"Found {len(checkpoints)} checkpoints. Using latest: {latest_checkpoint}")
        return latest_checkpoint

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
                add_generation_prompt=True,
                enable_thinking=False
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
            
            # Compute total tokens generated
            if tgps_show_var: 
                token_len += len(output_ids)

            # Remove thinking content using regex and handle special tokens
            # result = re.sub(r'<think>[\s\S]*?</think>', '', full_result, flags=re.DOTALL).strip()
            index = len(output_ids) - output_ids[::-1].index(151668) if 151668 in output_ids else 0
            
            # Decode the full result
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            batch_outs.append(content)
            
        if tgps_show_var:
            return batch_outs[0] if single_input else batch_outs, token_len, gen_time
        
        # Return single string if input was single string, otherwise return list
        return batch_outs[0] if single_input else batch_outs, None, None
        
if __name__ == "__main__":
    """
    Usage:
    ----------------------
    1. Move this file to the agents/ directory.
    2. Make corresponding changes in the import structure of answer_agent.py to make it use for answering questions.
    3. How to run as main file?
        a. python -m agents.answer_model2
        b. change the adapter_type as shown: `AAgent(adapter_type=None) # adapter_type can be "sft" or "grpo"`
    """

    # Single message (backward compatible)
    ans_agent = AAgent(adapter_type=None) # adapter_type can be "sft" or "grpo"
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