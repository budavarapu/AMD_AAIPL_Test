import os
import torch
import argparse
import re
import json
import time
from typing import Optional, List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback
)
from trl import SFTTrainer, GRPOConfig, GRPOTrainer
from peft import LoraConfig, PeftModel
import wandb

class BloodRelationsTrainer:
    """Unified trainer class for both SFT and GRPO training with inference capabilities."""
    
    def __init__(self, args):
        """Initialize the trainer with configuration arguments."""
        self.args = args
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None
        
        # Inference model cache
        self._inference_model = None
        self._inference_tokenizer = None
        
        # Constants
        self.reasoning_start = "<reasoning>"
        self.reasoning_end = "</reasoning>"
        self.solution_start = "<answer>"
        self.solution_end = "</answer>"
        
        self.system_prompt = f"""
        You are an expert in logical reasoning and complex problem-solving.
        Your task is to answer multiple-choice questions (MCQs) on the topic of "Blood Relations".
        Think about the problem and provide your working out.
        Place it between {self.reasoning_start} and {self.reasoning_end}.
        Then, provide your answer between {self.solution_start} and {self.solution_end}
        Make sure to follow the XML format strictly i.e. start with {self.reasoning_start} and end with {self.reasoning_end} for reasoning,
        and start with {self.solution_start} and end with {self.solution_end} for the answer.
        Your answer should be a single letter (A, B, C, or D) representing the correct option.
        Do not include any additional text outside of these tags.
        """
        
        # Setup environment
        self._setup_environment()
        
        # Display configuration
        self._display_config()
    
    def _setup_environment(self):
        """Setup environment variables and GPU configuration."""
        # GPU selection - parse gpu_ids from args
        gpu_ids = [int(x.strip()) for x in self.args.gpu_ids.split(',') if x.strip().isdigit()]
        if not gpu_ids:
            gpu_ids = [0]  # Default to GPU 0 if no valid IDs provided
        
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ','.join(map(str, gpu_ids)))
        
        # Add environment variables for distributed training
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        
        # ROCm optimization flags for vLLM (if using GRPO)
        if self.args.training_type == 'grpo':
            os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "0")
            os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")
        
        print(f"PyTorch detected number of available devices: {torch.cuda.device_count()}")
    
    def _display_config(self):
        """Display training configuration."""
        print("=" * 60)
        print(f"BLOOD RELATIONS {self.args.training_type.upper()} TRAINER")
        print("=" * 60)
        print(f"Training Type: {self.args.training_type}")
        print(f"Mode: {self.args.mode}")
        print(f"Model: {self.args.model_name}")
        print(f"Output directory: {self.args.output_dir}")
        print(f"Dataset file: {self.args.dataset_file}")
        print(f"Learning rate: {self.args.learning_rate}")
        print(f"Epochs: {self.args.num_train_epochs}")
        print(f"Batch size: {self.args.per_device_train_batch_size}")
        print(f"LoRA rank: {self.args.lora_r}")
        print(f"LoRA alpha: {self.args.lora_alpha}")
        print(f"Max sequence length: {self.args.max_seq_length}")
        print(f"GPU IDs: {self.args.gpu_ids}")
        print("=" * 60)
    
    def load_dataset(self) -> Dataset:
        """Load and process the blood relations dataset."""
        print(f"Loading dataset from: {self.args.dataset_file}")
        
        # Load raw data once
        raw_items = self._load_raw_dataset()
        
        # Format based on training type
        if self.args.training_type == 'sft':
            self.dataset = self._format_sft_dataset(raw_items)
        else:  # grpo
            self.dataset = self._format_grpo_dataset(raw_items)
    
    def _load_raw_dataset(self) -> List[Dict[str, str]]:
        """Load raw dataset items from JSON file."""
        items = []
        
        try:
            with open(self.args.dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both array format and single object format
            if isinstance(data, list):
                # Array of question objects
                question_objects = data
            else:
                # Single object, wrap in list
                question_objects = [data]
            
            for idx, question_obj in enumerate(question_objects, 1):
                try:
                    # Extract question from the new JSON format
                    if "question" in question_obj:
                        topic = question_obj.get("topic", "Blood Relations")

                        question_text = question_obj["question"]
                        
                        # Extract choices and format them as part of the question
                        choices = question_obj.get("choices", [])
                        if choices:
                            # Combine question with choices
                            full_question = f"Topic: {topic}\n" + f"Question: {question_text}\n" + "\n".join(choices)
                        else:
                            full_question = f"Topic: {topic}\n" + f"Question: {question_text}"
                        
                        # Extract answer
                        answer_letter = question_obj.get("answer", "").strip()
                        
                        # Extract explanation as reasoning
                        reasoning = question_obj.get("explanation", "Let me analyze this step by step.")
                        
                        if full_question and answer_letter:
                            items.append({
                                'question': full_question,
                                'answer': answer_letter,
                                'reasoning': reasoning,
                                'original_question': question_text,
                                'choices': choices
                            })
                            
                            if idx <= 3:  # Show first 3 for debugging
                                print(f"✓ Loaded question {idx}: {question_text[:50]}...")
                        else:
                            print(f"Warning: Incomplete question data at index {idx}")
                            
                except Exception as e:
                    print(f"Warning: Error processing question at index {idx}: {e}")
                    continue

        except FileNotFoundError:
            print(f"Error: Dataset file {self.args.dataset_file} not found.")
            raise
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in {self.args.dataset_file}: {e}")
            raise
        except Exception as e:
            print(f"Error: Unexpected error loading dataset: {e}")
            raise
    
        if not items:
            print("Error: No valid items found in dataset file.")
            raise ValueError("No valid training data found")
        
        print(f"Successfully loaded {len(items)} questions from JSON file")
        return items

    def _format_sft_dataset(self, raw_items: List[Dict[str, str]]) -> Dataset:
        """Format raw items for SFT training."""
        # Load tokenizer for formatting
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        formatted_items = []
        
        for item in raw_items:
            question = item['question']
            answer_letter = item['answer']
            reasoning = item['reasoning']
            
            # Compose the model's expected output
            model_completion = (
                f"{self.reasoning_start}"
                f"{reasoning}"
                f"{self.reasoning_end}"
                f"{self.solution_start}"
                f"{answer_letter}"
                f"{self.solution_end}"
            )
            
            # Format as chat
            chat_messages = [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': question},
                {'role': 'assistant', 'content': model_completion}
            ]
            
            # Apply chat template
            formatted_text = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            formatted_items.append({
                "text": formatted_text,
                "question": question,
                "answer": model_completion
            })
        
        print(f"Created {len(formatted_items)} SFT training samples")
        return Dataset.from_list(formatted_items)

    def _format_grpo_dataset(self, raw_items: List[Dict[str, str]]) -> Dataset:
        """Format raw items for GRPO training."""
        formatted_items = []
        
        for item in raw_items:
            question = item['question']
            answer_letter = item['answer']
            
            # Format for GRPO
            formatted_items.append({
                'prompt': [
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': question}
                ],
                'answer': answer_letter
            })
        
        print(f"Created {len(formatted_items)} GRPO training samples")
        return Dataset.from_list(formatted_items)
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer."""
        print(f"Loading model: {self.args.model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
        )

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
                
        # Setup model configuration
        self.model.config.use_cache = False
        if hasattr(self.model.config, 'pretraining_tp'):
            self.model.config.pretraining_tp = 1
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name, 
            trust_remote_code=True, 
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.padding_side = "right"
        self.tokenizer.model_max_length = self.args.max_seq_length
        
        print("Model and tokenizer loaded successfully")
    
    def setup_peft_config(self) -> LoraConfig:
        """Setup LoRA configuration."""
        return LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )
    
    def setup_wandb(self):
        """Initialize Weights & Biases logging."""
        if not self.args.disable_wandb:
            try:
                wandb.init(
                    project=self.args.wandb_project,
                    name=self.args.wandb_run_name,
                    config={
                        "training_type": self.args.training_type,
                        "model_name": self.args.model_name,
                        "learning_rate": self.args.learning_rate,
                        "num_train_epochs": self.args.num_train_epochs,
                        "per_device_train_batch_size": self.args.per_device_train_batch_size,
                        "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                        "lora_r": self.args.lora_r,
                        "lora_alpha": self.args.lora_alpha,
                        "lora_dropout": self.args.lora_dropout,
                        "max_seq_length": self.args.max_seq_length,
                        "dataset_file": self.args.dataset_file,
                        "gpu_ids": self.args.gpu_ids,
                    }
                )
                print("Weights & Biases initialized successfully")
            except Exception as e:
                print(f"WandB initialization failed: {e}. Training will continue without WandB.")
                self.args.disable_wandb = True
    
    def train_sft(self):
        """Train using Supervised Fine-Tuning."""
        print("Starting SFT training...")
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            save_strategy="steps",
            save_steps=100,
            logging_steps=10,
            learning_rate=self.args.learning_rate,
            weight_decay=0.01,
            fp16=False,
            bf16=True,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="wandb" if not self.args.disable_wandb else "none",
        )
        
        # Setup LoRA config
        peft_config = self.setup_peft_config()
        
        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,  # Changed from 'tokenizer' to 'processing_class'
            args=training_args,
            train_dataset=self.dataset,
            peft_config=peft_config,
            # dataset_text_field="text",
            # max_seq_length=self.args.max_seq_length,
        )
        
        # Start training
        self.trainer.train()
        
        # Save model
        print(f"Saving LoRA adapters to {self.args.output_dir}")
        self.trainer.save_model(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)
        
        print("SFT training completed successfully")
    
    def train_grpo(self):
        """Train using Group Relative Policy Optimization."""
        print("Starting GRPO training...")
        
        # Setup training arguments
        training_args = GRPOConfig(
            output_dir=self.args.output_dir,
            learning_rate=self.args.learning_rate,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_steps=100,
            lr_scheduler_type='cosine_with_restarts',
            logging_steps=1,
            bf16=True,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            num_generations=4,
            max_prompt_length=self.args.max_prompt_length,
            max_completion_length=self.args.max_seq_length - self.args.max_prompt_length,
            num_train_epochs=self.args.num_train_epochs,
            save_steps=100,
            log_on_each_node=False,
            use_vllm=True,
            vllm_gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
            vllm_mode="colocate",
            report_to="wandb" if not self.args.disable_wandb else "none",
            generation_kwargs={
                "temperature": self.args.temperature,
                "max_tokens": self.args.max_seq_length - self.args.max_prompt_length,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        )
        
        # Setup LoRA config
        peft_config = self.setup_peft_config()
        
        # Create trainer
        self.trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[self._combined_reward_func],
            args=training_args,
            train_dataset=self.dataset,
            peft_config=peft_config,
        )
        
        # Add callbacks
        self.trainer.add_callback(self._AdjustContextLengthCallback())
        
        # Start training
        self.trainer.train()
        
        print("GRPO training completed successfully")
    
    def _combined_reward_func(self, prompts, completions, answer, **kwargs) -> List[float]:
        """Combined reward function for GRPO training."""
        format_scores = self._format_reward_func(completions, **kwargs)
        correctness_scores = self._correctness_reward_func(prompts, completions, answer, **kwargs)
        length_penalty_scores = self._length_penalty_reward_func(completions, **kwargs)
        
        final_rewards = []
        num_samples = len(completions)

        for i in range(num_samples):
            format_score = format_scores[i]
            # raw_correctness_score = correctness_scores[i]
            length_penalty = length_penalty_scores[i]
            
            # Only apply correctness reward if format is correct
            actual_correctness_score = 0.0
            # if format_score > 0:
                # actual_correctness_score = raw_correctness_score
            
            total_reward = format_score + length_penalty + actual_correctness_score
            final_rewards.append(total_reward)
            
            if i == 0:  # Log first sample
                print(f"[REWARD] Format: {format_score:.3f}, "
                      f"Correctness: {actual_correctness_score:.3f}, "
                      f"Length: {length_penalty:.3f}, "
                      f"Total: {total_reward:.3f}")
        
        return final_rewards
    
    def _format_reward_func(self, completions, **kwargs) -> List[float]:
        """Reward function for format correctness."""
        pattern = r"(?s)^<reasoning>.*?</reasoning>\n<answer>.*?</answer>$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [1.0 if match else 0.0 for match in matches]
    
    def _correctness_reward_func(self, prompts, completions, answer, **kwargs) -> List[float]:
        """Reward function for answer correctness."""
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [self.extract_xml_answer(r) for r in responses]
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
    
    def _length_penalty_reward_func(self, completions, **kwargs) -> List[float]:
        """Reward function for response length."""
        responses = [completion[0]['content'] for completion in completions]
        rewards = []
        
        for response in responses:
            response_length = len(response)
            optimal_length = 512
            max_acceptable_length = 768
            
            if response_length <= optimal_length:
                length_reward = 0.5
            elif response_length <= max_acceptable_length:
                penalty_factor = (response_length - optimal_length) / (max_acceptable_length - optimal_length)
                length_reward = 0.5 * (1 - penalty_factor)
            else:
                excess_length = response_length - max_acceptable_length
                length_reward = -1.0 * (excess_length / 1000)
            
            rewards.append(length_reward)
        
        return rewards
    
    class _AdjustContextLengthCallback(TrainerCallback):
        """Callback to adjust context length during training."""
        
        def on_step_begin(self, args, state, control, **kwargs):
            step = state.global_step
            if step >= 1000:
                args.max_prompt_length = 384
            elif step >= 500:
                args.max_completion_length = 256
            
            if step in [500, 1000]:
                print(f"Adjusted context length at step {step}")
    
    def train(self):
        """Main training function that handles both SFT and GRPO."""
        print(f"Starting {self.args.training_type.upper()} training...")
        
        # Load dataset
        self.load_dataset()
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Setup wandb
        self.setup_wandb()
        
        # Train based on type
        if self.args.training_type == 'sft':
            self.train_sft()
        else:  # grpo
            self.train_grpo()
        
        # Cleanup
        if not self.args.disable_wandb and wandb.run:
            wandb.finish()
        
        print(f"{self.args.training_type.upper()} training completed!")
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in the output directory."""
        if not os.path.exists(self.args.output_dir):
            print(f"Output directory {self.args.output_dir} does not exist")
            return None
        
        checkpoint_pattern = re.compile(r'checkpoint-(\d+)')
        checkpoints = []
        
        for item in os.listdir(self.args.output_dir):
            item_path = os.path.join(self.args.output_dir, item)
            if os.path.isdir(item_path):
                match = checkpoint_pattern.match(item)
                if match:
                    step_num = int(match.group(1))
                    checkpoints.append((step_num, item_path))
        
        if not checkpoints:
            print(f"No checkpoints found in {self.args.output_dir}")
            return None
        
        checkpoints.sort(key=lambda x: x[0])
        latest_checkpoint = checkpoints[-1][1]
        
        print(f"Found {len(checkpoints)} checkpoints. Using latest: {latest_checkpoint}")
        return latest_checkpoint
    
    def extract_xml_answer(self, text: str) -> str:
        """Extract letter answer from XML tags."""
        if "<answer>" not in text or "</answer>" not in text:
            return ""
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        answer_letter = re.search(r'(?<![a-zA-Z])[A-D](?![a-zA-Z])|(?<![a-zA-Z])[A-D]\)|[A-D]\.', answer)
        if answer_letter:
            return answer_letter.group(0)[0].upper()
        return ""
    
    def setup_inference_model(self):
        """Setup model once for inference mode."""
        if hasattr(self, '_inference_model') and self._inference_model is not None:
            return  # Already loaded
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Setting up inference model on device: {device}")
            
            # Load base model
            print(f"Loading base model: {self.args.model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load tokenizer
            print(f"Loading tokenizer for: {self.args.model_name}")
            self._inference_tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code=True)
            if self._inference_tokenizer.pad_token is None:
                self._inference_tokenizer.pad_token = self._inference_tokenizer.eos_token
            self._inference_tokenizer.model_max_length = self.args.max_seq_length
            
            # Find and load checkpoint
            checkpoint_path = self.find_latest_checkpoint()
            if checkpoint_path is None:
                print(f"No trained checkpoints found in {self.args.output_dir}")
                self._inference_model = base_model
            else:
                print(f"Loading LoRA adapters from: {checkpoint_path}")
                self._inference_model = PeftModel.from_pretrained(base_model, checkpoint_path)
            
            self._inference_model = self._inference_model.eval()
            
            print("Inference model setup completed successfully")
            
        except Exception as e:
            error_message = f"Error setting up inference model: {str(e)}"
            print(error_message)
            raise

    def generate_response(self, prompt: str, sys_prompt: str = None) -> str:
        """
        Generate response using the trained model checkpoints.
        
        Args:
            prompt (str): The user question/prompt
            sys_prompt (str, optional): System prompt. If None, uses default.
        
        Returns:
            str: Generated response from the model
        """
        if sys_prompt is None:
            sys_prompt = self.system_prompt
        
        try:
            # Setup model if not already done
            self.setup_inference_model()
            
            # Prepare messages
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template
            full_prompt_text = self._inference_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            # Generate response
            inputs = self._inference_tokenizer(full_prompt_text, return_tensors="pt").to(self._inference_model.device)
            input_len = inputs["input_ids"].shape[-1]
            
            start_time = time.time()
            with torch.inference_mode():
                generation_output = self._inference_model.generate(
                    **inputs, 
                    max_new_tokens=768, 
                    do_sample=False, # For deterministic output
                    pad_token_id=self._inference_tokenizer.eos_token_id # Important for generation
                )
            end_time = time.time()
            
            generated_tokens = generation_output[0][input_len:]
            decoded_response = self._inference_tokenizer.decode(generated_tokens, skip_special_tokens=True)
           
            duration = end_time - start_time
            print(f"Response generated in {duration:.2f}s. Length: {len(decoded_response)} chars.")
            
            return decoded_response.strip()
            
        except Exception as e:
            error_message = f"Error during inference: {str(e)}"
            print(error_message)
            return f"ERROR: {error_message}"
    
    def batch_inference(self):
        """Process multiple questions from a file and save responses."""
        test_file = self.args.test_file or self.args.dataset_file
        output_file = self.args.inference_output
        
        print("="*60)
        print("RUNNING BATCH INFERENCE")
        print("="*60)
        print(f"Loading questions from: {test_file}")
        
        # Load test dataset from JSON
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
            # Handle both array format and single object format
            if isinstance(data, list):
                question_objects = data
            else:
                question_objects = [data]
        
            questions = []
            answers = []
        
            for question_obj in question_objects:
                try:
                    # Extract question text
                    question_text = question_obj.get("question", "")
                
                    # Extract choices and format them as part of the question
                    choices = question_obj.get("choices", [])
                    if choices:
                        # Combine question with choices
                        full_question = question_text + "\n" + "\n".join(choices)
                    else:
                        full_question = question_text
                
                    # Extract answer
                    answer_letter = question_obj.get("answer", "").strip()
                
                    if full_question and answer_letter:
                        questions.append(full_question)
                        answers.append(answer_letter)
                    else:
                        print(f"Warning: Incomplete question data in object: {question_obj}")
                    
                except Exception as e:
                    print(f"Warning: Error processing question object: {e}")
                    continue
        
            print(f"Successfully loaded {len(questions)} questions from JSON file")
        
        except FileNotFoundError:
            print(f"Error: Test file {test_file} not found.")
            return
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in {test_file}: {e}")
            return
        except Exception as e:
            print(f"Error loading test file: {e}")
            return

        if not questions:
            print("No valid questions found in test file.")
            return

        print(f"Processing {len(questions)} questions...")
        
        # Setup inference model once before processing all questions
        print("Setting up inference model...")
        try:
            self.setup_inference_model()
        except Exception as e:
            print(f"Failed to setup inference model: {e}")
            return

        with open(output_file, "w", encoding='utf-8') as f:
            f.write("# Batch Inference Results\n\n")
            f.write(f"Model: {self.args.model_name}\n")
            f.write(f"Training Type: {self.args.training_type}\n")
            f.write(f"Checkpoint: {self.find_latest_checkpoint()}\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            correct_count = 0
            format_correct_count = 0  # Count of responses with extractable answers
            total_count = len(questions)
            
            # Process all questions with the same loaded model
            for idx, (question, expected_answer) in enumerate(zip(questions, answers), 1):
                print(f"Processing question {idx}/{total_count}...")
            
                response = self.generate_response(question)
                extracted_answer = self.extract_xml_answer(response)
                
                # Check if answer was extractable (format correctness)
                is_format_correct = extracted_answer != ""
                if is_format_correct:
                    format_correct_count += 1
                
                # Check if answer is correct
                is_correct = extracted_answer == expected_answer
                if is_correct:
                    correct_count += 1
            
                f.write(f"## Question {idx}\n\n")
                f.write(f"**Question:** {question}\n\n")
                f.write(f"**Expected Answer:** {expected_answer}\n\n")
                f.write(f"**Model Response:**\n```\n{response}\n```\n\n")
                f.write(f"**Extracted Answer:** {extracted_answer if extracted_answer else 'N/A (Format Error)'}\n\n")
                f.write(f"**Format Correct:** {'✅' if is_format_correct else '❌'}\n\n")
                f.write(f"**Answer Correct:** {'✅' if is_correct else '❌'}\n\n")
                f.write("---\n\n")

            # Calculate percentages
            accuracy = correct_count / total_count * 100
            format_accuracy = format_correct_count / total_count * 100
            
            # Write summary
            f.write(f"## Summary\n\n")
            f.write(f"Total Questions: {total_count}\n")
            f.write(f"Correct Answers: {correct_count}\n")
            f.write(f"Format Correct: {format_correct_count}\n")
            f.write(f"Answer Accuracy: {accuracy:.2f}%\n")
            f.write(f"Format Accuracy: {format_accuracy:.2f}%\n")
            
        print(f"Batch inference complete. Results saved to {output_file}")
        print(f"Answer Accuracy: {correct_count}/{total_count} ({accuracy:.2f}%)")
        print(f"Format Accuracy: {format_correct_count}/{total_count} ({format_accuracy:.2f}%)")
            
    def cleanup_inference_model(self):
        """Clean up inference model to free memory."""
        if hasattr(self, '_inference_model') and self._inference_model is not None:
            del self._inference_model
            self._inference_model = None
        if hasattr(self, '_inference_tokenizer') and self._inference_tokenizer is not None:
            del self._inference_tokenizer
            self._inference_tokenizer = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Inference model cleaned up")

    def test_single_inference(self):
        """Test inference with a single question."""
        if not hasattr(self.args, 'test_question') or not self.args.test_question:
            print("No test question provided. Use --test_question argument.")
            return
        
        print("="*60)
        print("RUNNING SINGLE INFERENCE TEST")
        print("="*60)
        print(f"Question: {self.args.test_question}")
        
        # Setup inference model
        print("Setting up inference model...")
        try:
            self.setup_inference_model()
        except Exception as e:
            print(f"Failed to setup inference model: {e}")
            return
        
        # Generate response
        print("Generating response...")
        response = self.generate_response(self.args.test_question)
        extracted_answer = self.extract_xml_answer(response)
        
        # Display results
        print("\n" + "="*60)
        print("INFERENCE RESULTS")
        print("="*60)
        print(f"Question: {self.args.test_question}")
        print(f"\nModel Response:\n{response}")
        print(f"\nExtracted Answer: {extracted_answer}")
        print("="*60)
        
        # Save results if output file specified
        if hasattr(self.args, 'inference_output') and self.args.inference_output:
            with open(self.args.inference_output, "w", encoding='utf-8') as f:
                f.write("# Single Inference Test Results\n\n")
                f.write(f"Model: {self.args.model_name}\n")
                f.write(f"Training Type: {self.args.training_type}\n")
                f.write(f"Checkpoint: {self.find_latest_checkpoint()}\n")
                f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Question:** {self.args.test_question}\n\n")
                f.write(f"**Model Response:**\n```\n{response}\n```\n\n")
                f.write(f"**Extracted Answer:** {extracted_answer}\n")
            
            print(f"Results saved to: {self.args.inference_output}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Blood Relations Trainer")
    
    # General arguments
    parser.add_argument("--model_name", type=str, default="/jupyter-tutorial/hf_models/Llama-3.2-1B-Instruct", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="checkpoints/demo/sft", help="Output directory for checkpoints and model")
    parser.add_argument("--dataset_file", type=str, default="formatted_questions_array.json", help="Path to the dataset file (JSON)")
    parser.add_argument("--test_file", type=str, default="test_questions_array.json", help="Path to the test file for inference")
    parser.add_argument("--test_question", type=str, help="Single question for testing inference")
    parser.add_argument("--inference_output", type=str, default="inference.md", help="Output file for inference results")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs to use (comma-separated)")
    
    # Training arguments
    parser.add_argument("--training_type", type=str, choices=["sft", "grpo"], default="grpo", help="Type of training")
    parser.add_argument("--mode", type=str, choices=["train", "inference", "both"], default="train", help="Mode of operation")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Maximum prompt length for GRPO")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for response generation")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9, help="vLLM GPU memory utilization")
    
    # WandB arguments
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="blood_relations", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, help="WandB run name")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Auto-generate wandb run name if not provided
    model_display_name = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{model_display_name}-{args.training_type}-r{args.lora_r}-lr{args.learning_rate}"
    
    # Create trainer
    trainer = BloodRelationsTrainer(args)
    
    try:
        # Run based on mode
        if args.mode == 'train':
            trainer.train()
        
        elif args.mode == 'inference':
            if args.test_question:
                trainer.test_single_inference()
            else:
                trainer.batch_inference()
        
        elif args.mode == 'both':
            # Train first
            trainer.train()
            print("\n" + "="*60)
            print("TRAINING COMPLETED - STARTING INFERENCE")
            print("="*60)
            
            # Run inference (batch or single based on what's provided)
            if args.test_question:
                trainer.test_single_inference()
            else:
                trainer.batch_inference()
    
    except Exception as e:
        print(f"Error during processing: {e}")
    
    finally:
        # Cleanup
        if not args.disable_wandb and wandb.run:
            wandb.finish()
        trainer.cleanup_inference_model()

if __name__ == "__main__":
    """
    BLOOD RELATIONS UNIFIED TRAINER
    ================================

    This script provides a unified interface for training blood relations reasoning models
    using either Supervised Fine-Tuning (SFT) or Group Relative Policy Optimization (GRPO).
    It also includes comprehensive inference capabilities.

    USAGE EXAMPLES:
    1. SFT TRAINING:

       python -m trainer \
           --training_type sft \
           --mode train \
           --model_name /jupyter-tutorial/hf_models/Llama-3.2-1B-Instruct \
           --output_dir checkpoints/demo/sft \
           --learning_rate 2e-5 \
           --num_train_epochs 3 \
           --per_device_train_batch_size 4 \
           --lora_r 32 \
           --lora_alpha 64

   

    2. GRPO TRAINING:

       python -m trainer \
           --training_type grpo \
           --mode train \
           --model_name /jupyter-tutorial/hf_models/Llama-3.2-1B-Instruct \
           --output_dir checkpoints/demo/grpo \
           --learning_rate 1e-5 \
           --num_train_epochs 2 \
           --per_device_train_batch_size 2 \
           --gradient_accumulation_steps 2 \
           --lora_r 16 \
           --lora_alpha 32 \
           --vllm_gpu_memory_utilization 0.7

   

    3. INFERENCE ONLY:

       python -m trainer \
           --mode inference \
           --model_name /jupyter-tutorial/hf_models/Llama-3.2-1B-Instruct \
           --output_dir checkpoints/demo/sft \
           --test_question "If A is B's father and B is C's mother, what is A to C?"

   

    4. TRAIN + INFERENCE:

       python -m trainer \
           --training_type sft \
           --mode both \
           --model_name /jupyter-tutorial/hf_models/Llama-3.2-1B-Instruct \
           --output_dir checkpoints/demo/sft \
           --test_question "If A is B's father and B is C's mother, what is A to C?"

   

    5. BATCH INFERENCE:

       python -m trainer \
           --mode inference \
           --model_name /jupyter-tutorial/hf_models/Llama-3.2-1B-Instruct \
           --output_dir checkpoints/demo/sft \
           --test_file test_questions.txt \
           --inference_output batch_results.md

   

    6. MULTI-GPU TRAINING:

       python -m trainer \
           --training_type sft \
           --mode train \
           --model_name /jupyter-tutorial/hf_models/Llama-3.1-8B-Instruct \
           --gpu_ids "0,1,2,3" \
           --per_device_train_batch_size 2 \
           --gradient_accumulation_steps 4

   

    7. DISABLE WANDB LOGGING:

       python -m trainer \
           --training_type sft \
           --mode train \
           --disable_wandb

   

    KEY FEATURES:
    - Unified SFT and GRPO training in a single file
    - LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
    - Built-in inference capabilities with model card-like interface
    - Comprehensive reward functions for GRPO training
    - WandB integration for experiment tracking
    - Multi-GPU support
    - Batch inference with accuracy evaluation
    - Structured XML output format for reasoning and answers

    GRPO SPECIFIC FEATURES:
    - Custom reward functions for format, correctness, and length
    - Dynamic context length adjustment during training
    - vLLM integration for efficient generation
    - Support for multiple generations per prompt
   
    INFERENCE FEATURES:
    - Single question inference for quick testing
    - Batch inference for evaluation on test sets
    - Automatic checkpoint detection and loading
    - Structured response extraction and evaluation
    - Markdown output for easy result review
    """
    main()