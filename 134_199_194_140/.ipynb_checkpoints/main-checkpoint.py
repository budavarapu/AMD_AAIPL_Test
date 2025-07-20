#!/usr/bin/env python3
"""
🏆 HACKATHON COMPETITION CLI TOOL 🏆
Fixed version with JSON parsing and no thinking
"""

import os
import json
import time
import torch
import re
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
from datetime import datetime

class CompetitionAgent:
    """Unified agent for both question generation and answer solving"""
    
    def __init__(self, agent_type: str, model_path: str = "/home/user/hf_models/Qwen3-4B"):
        self.agent_type = agent_type  # "question" or "answer"
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.token_limit = 100
        
        # Load appropriate checkpoint
        if agent_type == "question":
            self.checkpoint_dir = "checkpoints/question_agent_sft"
        else:
            self.checkpoint_dir = "checkpoints/answer_agent_sft"
        
        self.system_prompts = self._load_system_prompts()
        
    def _load_system_prompts(self) -> Dict[str, str]:
        """Load optimized system prompts"""
        if self.agent_type == "question":
            return {
                "default": """Generate EXTREMELY DIFFICULT Number Series MCQ under 100 tokens.
Format: {"topic":"Number Series","question":"Next: a,b,c,d,?","choices":["A) num","B) num","C) num","D) num"],"answer":"A","explanation":"pattern explanation"}
Use complex patterns: nested sequences, prime operations, factorial combinations.
Make it championship-level difficulty (only 20-30% can solve).
CRITICAL: Output ONLY valid JSON, no thinking tags, no extra text."""
            }
        else:  # answer agent
            return {
                "default": """You are a championship-level MCQ solver specializing in Number Series.
Analyze the pattern step-by-step and provide precise answer.
CRITICAL: Output ONLY valid JSON format: {"answer":"A","reasoning":"pattern explanation under 80 words"}
No thinking tags, no extra text."""
            }
    
    def load_model(self):
        """Load the trained model"""
        if self.model is not None:
            return  # Already loaded
        
        print(f"🚀 Loading {self.agent_type.title()} Agent...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load LoRA adapters if available
            if os.path.exists(self.checkpoint_dir):
                print(f"✅ Loading trained checkpoint from {self.checkpoint_dir}")
                self.model = PeftModel.from_pretrained(base_model, self.checkpoint_dir)
            else:
                print(f"⚠️  No checkpoint found at {self.checkpoint_dir}, using base model")
                self.model = base_model
            
            self.model.eval()
            print(f"✅ {self.agent_type.title()} Agent loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON from model response, handling thinking tags and extra text"""
        # Remove thinking tags
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)
        
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Try to find JSON object
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, try to extract key components
        print(f"⚠️  No valid JSON found. Raw response: {response[:200]}...")
        
        # Try to extract answer and create JSON
        if self.agent_type == "answer":
            # Look for answer pattern
            answer_match = re.search(r'["\']?answer["\']?\s*:\s*["\']?([A-D])["\']?', response, re.IGNORECASE)
            reasoning_match = re.search(r'["\']?reasoning["\']?\s*:\s*["\']([^"\']+)["\']', response, re.IGNORECASE)
            
            if answer_match:
                return {
                    "answer": answer_match.group(1).upper(),
                    "reasoning": reasoning_match.group(1) if reasoning_match else "Pattern analysis"
                }
        
        # Return error dict
        return {"error": "Failed to parse JSON", "raw_response": response[:500]}
    
    def validate_question_tokens(self, question_data: Dict) -> tuple[bool, int]:
        """Validate question meets token limit"""
        if self.agent_type != "question" or "error" in question_data:
            return True, 0
        
        # Count core tokens (excluding explanation)
        core_content = f"{question_data.get('topic', '')} {question_data.get('question', '')} {' '.join(question_data.get('choices', []))} {question_data.get('answer', '')}"
        token_count = self.count_tokens(core_content)
        
        return token_count <= self.token_limit, token_count
    
    def generate_question(self, difficulty: str = "championship") -> Dict:
        """Generate a competition-quality Number Series question"""
        if self.agent_type != "question":
            raise ValueError("This agent is not configured for question generation")
        
        self.load_model()
        
        # Use optimized prompt for Number Series
        system_prompt = self.system_prompts["default"]
        user_prompt = f"Create an extremely difficult Number Series MCQ with complex integer pattern. Championship difficulty."
        
        # Generate question
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._generate_response(messages, max_tokens=400)
        
        # Extract JSON from response
        question_data = self.extract_json_from_response(response)
        
        if "error" not in question_data:
            # Validate token limit
            is_valid, token_count = self.validate_question_tokens(question_data)
            question_data["token_count"] = token_count
            question_data["token_valid"] = is_valid
            
            if not is_valid:
                print(f"⚠️  Token limit exceeded: {token_count}/100")
        
        return question_data
    
    def solve_question(self, question_data: Dict) -> Dict:
        """Solve a given question"""
        if self.agent_type != "answer":
            raise ValueError("This agent is not configured for answer solving")
        
        self.load_model()
        
        # Get system prompt
        system_prompt = self.system_prompts["default"]
        
        # Format question for solving (matching answer_agent.py format)
        question_text = question_data.get("question", "")
        choices = question_data.get("choices", [])
        
        # Use the exact format from answer_agent.py
        user_prompt = f"""INSTRUCTIONS FOR ANSWERING:
1. Carefully read and understand what is being asked.
2. Consider why each choice might be correct or incorrect.
3. There is only **ONE OPTION** correct.
4. Provide reasoning within 100 words

Now answer the following question:
Question: {question_text}
Choices: {' '.join(choices)}

RESPONSE FORMAT: Strictly generate a valid JSON object as shown below:
{{
    "answer": "One of the letter from [A, B, C, D]",
    "reasoning": "Brief explanation within 100 words"
}}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._generate_response(messages, max_tokens=200)
        
        # Extract JSON from response
        answer_data = self.extract_json_from_response(response)
        return answer_data
    
    def _generate_response(self, messages: List[Dict], max_tokens: int = 400) -> str:
        """Generate response using the model"""
        # Apply chat template - DISABLE THINKING
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # CRITICAL: Disable thinking tags
        )
        
        # Tokenize and generate
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.8,  # Slightly higher for creativity
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return response.strip()

class CompetitionCLI:
    """Simplified CLI for the competition system"""
    
    def __init__(self):
        self.question_agent = None
        self.answer_agent = None
        self.session_log = []
    
    def display_banner(self):
        """Display competition banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                  🏆 HACKATHON COMPETITION TOOL 🏆           ║
║                                                              ║
║  🎯 Question Agent: Generate championship-level MCQs        ║
║  🧠 Answer Agent: Solve with perfect accuracy               ║
║  ⚡ Token Optimized: <100 tokens per question               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(banner)
    
    def main_menu(self):
        """Display main menu and handle user choice"""
        while True:
            print("\n" + "="*60)
            print("🚀 COMPETITION MENU")
            print("="*60)
            print("1. 🎯 Question Agent - Generate MCQ")
            print("2. 🧠 Answer Agent - Solve MCQ")
            print("3. 🔄 Battle Mode - Q vs A Agent")
            print("4. 📊 Session Statistics")
            print("5. 🔧 Agent Status")
            print("6. 📝 Save Session Log")
            print("7. ❌ Exit")
            print("="*60)
            
            choice = input("🎮 Enter your choice (1-7): ").strip()
            
            if choice == "1":
                self.question_mode()
            elif choice == "2":
                self.answer_mode()
            elif choice == "3":
                self.battle_mode()
            elif choice == "4":
                self.show_statistics()
            elif choice == "5":
                self.show_agent_status()
            elif choice == "6":
                self.save_session_log()
            elif choice == "7":
                print("👋 Goodbye! Good luck in the competition!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def question_mode(self):
        """Question generation mode - simplified to Number Series only"""
        print("\n🎯 QUESTION GENERATION MODE")
        print("-" * 40)
        print("Generating championship-level Number Series question...")
        
        # Initialize question agent if needed
        if self.question_agent is None:
            self.question_agent = CompetitionAgent("question")
        
        start_time = time.time()
        
        try:
            question = self.question_agent.generate_question("championship")
            generation_time = time.time() - start_time
            
            print(f"\n✅ Question generated in {generation_time:.2f}s")
            print("="*60)
            
            if "error" in question:
                print(f"❌ Error: {question['error']}")
                print(f"Raw response: {question.get('raw_response', 'N/A')[:200]}...")
            else:
                self._display_question(question)
                
                # Log the question
                self.session_log.append({
                    "type": "question_generated",
                    "timestamp": datetime.now().isoformat(),
                    "question": question,
                    "generation_time": generation_time
                })
                
        except Exception as e:
            print(f"❌ Error generating question: {e}")
            import traceback
            traceback.print_exc()
    
    def answer_mode(self):
        """Answer solving mode"""
        print("\n🧠 ANSWER SOLVING MODE")
        print("-" * 40)
        
        # Initialize answer agent if needed
        if self.answer_agent is None:
            self.answer_agent = CompetitionAgent("answer")
        
        print("Choose input method:")
        print("1. Manual input")
        print("2. Load from recent question")
        
        input_choice = input("Select method (1-2): ").strip()
        
        if input_choice == "1":
            question_data = self._get_manual_question()
        elif input_choice == "2":
            question_data = self._get_recent_question()
        else:
            print("❌ Invalid choice")
            return
        
        if not question_data:
            return
        
        print(f"\n🔍 Solving question...")
        start_time = time.time()
        
        try:
            answer = self.answer_agent.solve_question(question_data)
            solving_time = time.time() - start_time
            
            print(f"\n✅ Question solved in {solving_time:.2f}s")
            print("="*60)
            
            if "error" in answer:
                print(f"❌ Error: {answer['error']}")
                print(f"Raw response: {answer.get('raw_response', 'N/A')[:200]}...")
            else:
                print(f"🎯 Answer: {answer.get('answer', 'N/A')}")
                print(f"💭 Reasoning: {answer.get('reasoning', 'N/A')}")
                
                # Check if answer is correct (if we have the expected answer)
                correct = None
                if "answer" in question_data:
                    expected = question_data["answer"].upper()
                    given = answer.get("answer", "").upper()
                    correct = expected == given
                    print(f"✅ Correctness: {'Correct' if correct else 'Incorrect'} (Expected: {expected})")
                
                # Log the answer
                self.session_log.append({
                    "type": "question_solved",
                    "timestamp": datetime.now().isoformat(),
                    "question": question_data,
                    "answer": answer,
                    "solving_time": solving_time,
                    "correct": correct
                })
                
        except Exception as e:
            print(f"❌ Error solving question: {e}")
            import traceback
            traceback.print_exc()
    
    def battle_mode(self):
        """Battle mode - Q agent vs A agent"""
        print("\n🔄 BATTLE MODE - Q AGENT vs A AGENT")
        print("-" * 50)
        
        # Initialize both agents
        if self.question_agent is None:
            self.question_agent = CompetitionAgent("question")
        if self.answer_agent is None:
            self.answer_agent = CompetitionAgent("answer")
        
        num_rounds = int(input("Enter number of battle rounds (1-10): ") or "3")
        
        results = {"correct": 0, "total": 0, "avg_time": 0, "token_violations": 0}
        
        for round_num in range(1, num_rounds + 1):
            print(f"\n🥊 ROUND {round_num}/{num_rounds}")
            print("-" * 30)
            
            # Q-agent generates question
            print("🎯 Q-Agent generating question...")
            question = self.question_agent.generate_question("championship")
            
            if "error" in question:
                print("❌ Q-Agent failed to generate valid question")
                continue
            
            print("✅ Question generated!")
            self._display_question(question, show_answer=False)
            
            # Check token limit
            if not question.get("token_valid", True):
                print("⚠️  Token limit violation!")
                results["token_violations"] += 1
            
            # A-agent solves question  
            print("\n🧠 A-Agent solving...")
            start_time = time.time()
            answer = self.answer_agent.solve_question(question)
            solve_time = time.time() - start_time
            
            if "error" in answer:
                print("❌ A-Agent failed to solve")
                continue
            
            # Check correctness
            expected = question.get("answer", "").upper()
            given = answer.get("answer", "").upper()
            correct = expected == given
            
            print(f"🎯 A-Agent answered: {given}")
            print(f"✅ Expected: {expected}")
            print(f"⏱️  Time: {solve_time:.2f}s")
            print(f"🏆 Result: {'CORRECT' if correct else 'INCORRECT'}")
            
            # Update results
            results["total"] += 1
            if correct:
                results["correct"] += 1
            results["avg_time"] += solve_time
            
            # Log battle round
            self.session_log.append({
                "type": "battle_round",
                "timestamp": datetime.now().isoformat(),
                "round": round_num,
                "question": question,
                "answer": answer,
                "correct": correct,
                "solve_time": solve_time
            })
        
        # Display battle results
        if results["total"] > 0:
            accuracy = (results["correct"] / results["total"]) * 100
            avg_time = results["avg_time"] / results["total"]
            
            print(f"\n🏆 BATTLE RESULTS")
            print("="*40)
            print(f"Accuracy: {results['correct']}/{results['total']} ({accuracy:.1f}%)")
            print(f"Average Time: {avg_time:.2f}s")
            print(f"Token Violations: {results['token_violations']}/{results['total']}")
            print(f"Performance: {'🔥 EXCELLENT' if accuracy >= 70 else '⚡ GOOD' if accuracy >= 50 else '💪 NEEDS WORK'}")
    
    def show_statistics(self):
        """Show session statistics"""
        print("\n📊 SESSION STATISTICS")
        print("="*50)
        
        if not self.session_log:
            print("No activity in this session yet.")
            return
        
        questions_generated = len([x for x in self.session_log if x["type"] == "question_generated"])
        questions_solved = len([x for x in self.session_log if x["type"] == "question_solved"])
        battle_rounds = len([x for x in self.session_log if x["type"] == "battle_round"])
        
        print(f"Questions Generated: {questions_generated}")
        print(f"Questions Solved: {questions_solved}")
        print(f"Battle Rounds: {battle_rounds}")
        
        # Calculate accuracy if we have solved questions
        solved_entries = [x for x in self.session_log if x["type"] in ["question_solved", "battle_round"] and x.get("correct") is not None]
        if solved_entries:
            correct_answers = len([x for x in solved_entries if x["correct"]])
            accuracy = (correct_answers / len(solved_entries)) * 100
            print(f"Answer Accuracy: {correct_answers}/{len(solved_entries)} ({accuracy:.1f}%)")
        
        # Show recent activity
        print(f"\nRecent Activity:")
        for log_entry in self.session_log[-5:]:
            timestamp = log_entry["timestamp"].split("T")[1][:8]
            print(f"  {timestamp} - {log_entry['type'].replace('_', ' ').title()}")
    
    def show_agent_status(self):
        """Show agent loading status"""
        print("\n🔧 AGENT STATUS")
        print("="*40)
        
        q_status = "✅ Loaded" if self.question_agent and self.question_agent.model else "❌ Not Loaded"
        a_status = "✅ Loaded" if self.answer_agent and self.answer_agent.model else "❌ Not Loaded"
        
        print(f"Question Agent: {q_status}")
        print(f"Answer Agent: {a_status}")
        
        if self.question_agent:
            q_checkpoint = "✅ Found" if os.path.exists(self.question_agent.checkpoint_dir) else "❌ Missing"
            print(f"Q-Agent Checkpoint: {q_checkpoint} ({self.question_agent.checkpoint_dir})")
            
        if self.answer_agent:
            a_checkpoint = "✅ Found" if os.path.exists(self.answer_agent.checkpoint_dir) else "❌ Missing"
            print(f"A-Agent Checkpoint: {a_checkpoint} ({self.answer_agent.checkpoint_dir})")
    
    def save_session_log(self):
        """Save session log to file"""
        if not self.session_log:
            print("No activity to save.")
            return
        
        filename = f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.session_log, f, indent=2)
        
        print(f"✅ Session log saved to {filename}")
    
    def _display_question(self, question: Dict, show_answer: bool = True):
        """Display a formatted question"""
        print(f"📝 Topic: {question.get('topic', 'N/A')}")
        print(f"❓ Question: {question.get('question', 'N/A')}")
        
        choices = question.get('choices', [])
        for choice in choices:
            print(f"   {choice}")
        
        if show_answer:
            print(f"✅ Answer: {question.get('answer', 'N/A')}")
            print(f"💡 Explanation: {question.get('explanation', 'N/A')}")
        
        if question.get('token_count'):
            status = "✅" if question.get('token_valid') else "❌"
            print(f"🔢 Tokens: {question['token_count']}/100 {status}")
    
    def _get_manual_question(self) -> Dict:
        """Get question details from manual input"""
        print("\n📝 Enter question details:")
        
        topic = input("Topic [Number Series]: ").strip() or "Number Series"
        question_text = input("Question: ").strip()
        
        choices = []
        for i, letter in enumerate(['A', 'B', 'C', 'D']):
            choice = input(f"Choice {letter}: ").strip()
            if not choice.startswith(f"{letter})"):
                choice = f"{letter}) {choice}"
            choices.append(choice)
        
        answer = input("Correct answer (A/B/C/D): ").strip().upper()
        
        return {
            "topic": topic,
            "question": question_text,
            "choices": choices,
            "answer": answer
        }
    
    def _get_recent_question(self) -> Optional[Dict]:
        """Get a recently generated question"""
        recent_questions = [x for x in self.session_log if x["type"] == "question_generated"]
        
        if not recent_questions:
            print("❌ No recent questions found. Generate a question first.")
            return None
        
        print("\nRecent questions:")
        for i, log_entry in enumerate(recent_questions[-5:], 1):
            q = log_entry["question"]
            print(f"{i}. {q.get('question', 'N/A')[:50]}...")
        
        try:
            choice = int(input("Select question (1-5): ")) - 1
            if 0 <= choice < min(5, len(recent_questions)):
                return recent_questions[-(5-choice)]["question"]
            else:
                print("❌ Invalid selection")
                return None
        except ValueError:
            print("❌ Invalid input")
            return None
    
    def run(self):
        """Main entry point"""
        self.display_banner()
        self.main_menu()

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="🏆 Hackathon Competition CLI Tool")
    parser.add_argument("--test", action="store_true", help="Run quick system test")
    
    args = parser.parse_args()
    
    if args.test:
        run_system_test()
    else:
        # Run interactive CLI
        cli = CompetitionCLI()
        cli.run()

def run_system_test():
    """Run comprehensive system test"""
    print("🧪 Running System Test...")
    print("="*50)
    
    try:
        # Test Question Agent
        print("1. Testing Question Agent...")
        q_agent = CompetitionAgent("question")
        question = q_agent.generate_question("championship")
        
        if "error" in question:
            print(f"❌ Question Agent Failed: {question['error']}")
            return
        else:
            print("✅ Question Agent Working")
            print(f"   Generated: {question.get('question', 'N/A')[:50]}...")
            print(f"   Token Count: {question.get('token_count', 'N/A')}/100")
        
        # Test Answer Agent
        print("\n2. Testing Answer Agent...")
        a_agent = CompetitionAgent("answer")
        answer = a_agent.solve_question(question)
        
        if "error" in answer:
            print(f"❌ Answer Agent Failed: {answer['error']}")
            return
        else:
            print("✅ Answer Agent Working")
            print(f"   Answer: {answer.get('answer', 'N/A')}")
            print(f"   Reasoning: {answer.get('reasoning', 'N/A')[:50]}...")
        
            # Check correctness
            expected = question.get("answer", "").upper()
            given = answer.get("answer", "").upper()
            correct = expected == given
            
            print(f"\n3. System Integration Test:")
            print(f"   Expected: {expected}")
            print(f"   Generated: {given}")
            print(f"   Result: {'✅ PASS' if correct else '❌ FAIL'}")
            
            # Performance summary
            print(f"\n🏆 System Status: {'🔥 READY FOR COMPETITION' if correct else '⚠️ NEEDS ATTENTION'}")
        
    except Exception as e:
        print(f"❌ System Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()  # Properly indented call to main()