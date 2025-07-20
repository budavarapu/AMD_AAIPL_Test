#!/usr/bin/env python3

import argparse
import json
import os
import time
import yaml
import sys
from typing import List, Dict, Any

# Add current directory to path to import question_model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from question_model import QuestionModel
except ImportError as e:
    print(f"Error importing question_model: {e}")
    print("Make sure question_model.py is in the agents/ directory")
    sys.exit(1)

def load_config(config_path: str = "qgen.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    default_config = {
        "max_tokens": 1024,
        "temperature": 0.7,
        "model_path": "../hf_models/Qwen/Qwen2.5-Coder-3B-Instruct",
        "min_valid_percentage": 40.0
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            return config
        else:
            print(f"Config file {config_path} not found, using defaults")
            return default_config
    except Exception as e:
        print(f"Error loading config: {e}, using defaults")
        return default_config

def validate_question(question: Dict[str, Any]) -> bool:
    """Validate question format according to requirements"""
    try:
        required_fields = ['topic', 'question', 'choices', 'answer', 'explanation']
        
        # Check required fields
        for field in required_fields:
            if field not in question:
                return False
        
        # Check choices format
        choices = question['choices']
        if not isinstance(choices, list) or len(choices) != 4:
            return False
        
        # Check choice labels (A), B), C), D))
        for i, choice in enumerate(choices):
            expected_prefix = f"{chr(65+i)})"
            if not isinstance(choice, str) or not choice.strip().startswith(expected_prefix):
                return False
        
        # Check answer format
        answer = question['answer']
        if not isinstance(answer, str) or answer.strip() not in ['A', 'B', 'C', 'D']:
            return False
        
        # Check that fields are not empty
        if not question['question'].strip() or not question['explanation'].strip():
            return False
        
        return True
        
    except Exception:
        return False

def filter_questions(questions: List[Dict[str, Any]], verbose: bool = False) -> List[Dict[str, Any]]:
    """Filter questions to keep only valid ones"""
    valid_questions = []
    invalid_questions = []
    
    for i, question in enumerate(questions):
        if validate_question(question):
            valid_questions.append(question)
            if verbose:
                print(f"✓ Question {i+1} is valid")
        else:
            invalid_questions.append((i, question))
            if verbose:
                print(f"✗ Question {i+1} is invalid")
                print(f"  Question: {question}")
    
    print(f"\n=== Question Validation Results ===")
    print(f"Valid questions: {len(valid_questions)}/{len(questions)}")
    print(f"Invalid questions: {len(invalid_questions)}")
    
    if invalid_questions and verbose:
        print(f"\nFirst few invalid questions:")
        for i, (idx, q) in enumerate(invalid_questions[:3]):
            print(f"  {idx+1}: {q}")
    
    return valid_questions

def save_questions(questions: List[Dict[str, Any]], output_file: str, verbose: bool = False):
    """Save questions to JSON file"""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(questions, f, indent=2)
    
    print(f"Saved {len(questions)} questions to {output_file}")
    
    if verbose and questions:
        print(f"\n=== Sample Questions ===")
        for i, question in enumerate(questions[:2]):  # Show first 2 questions
            print(f"\nQuestion {i+1}:")
            print(json.dumps(question, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Generate questions using Q-Agent")
    parser.add_argument("--output_file", type=str, default="outputs/questions.json",
                       help="Output file path for generated questions")
    parser.add_argument("--num_questions", type=int, default=20,
                       help="Number of questions to generate")
    parser.add_argument("--config", type=str, default="qgen.yaml",
                       help="Configuration file path")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Override model path from config")
    
    args = parser.parse_args()
    
    print("="*50)
    print("AMD AI Premier League - Question Agent")
    print("="*50)
    print(f"Generating {args.num_questions} questions...")
    print(f"Output file: {args.output_file}")
    
    # Load configuration
    config = load_config(args.config)
    if args.model_path:
        config["model_path"] = args.model_path
    
    if args.verbose:
        print(f"\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # Initialize model
    try:
        print(f"\nInitializing Question Model...")
        start_time = time.time()
        
        question_model = QuestionModel(
            model_path=config["model_path"]
        )
        
        init_time = time.time() - start_time
        print(f"Model initialized in {init_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Generate questions
    try:
        print(f"\n🔄 Generating questions...")
        start_time = time.time()
        
        questions = question_model.generate_questions(
            num_questions=args.num_questions,
            max_tokens=config["max_tokens"]
        )
        
        generation_time = time.time() - start_time
        avg_time = generation_time / len(questions) if questions else 0
        
        print(f"\n⏱️  Generation completed:")
        print(f"  Total time: {generation_time:.2f} seconds")
        print(f"  Average per question: {avg_time:.2f} seconds")
        
        # Validate questions
        print(f"\n🔍 Validating questions...")
        valid_questions = filter_questions(questions, args.verbose)
        
        # Check success rate
        valid_percentage = (len(valid_questions) / args.num_questions * 100) if args.num_questions > 0 else 0
        print(f"Success rate: {valid_percentage:.1f}%")
        
        # Check if meets minimum requirement
        min_valid = config.get("min_valid_percentage", 40.0)
        if valid_percentage < min_valid:
            print(f"⚠️  WARNING: Success rate {valid_percentage:.1f}% is below minimum {min_valid}%")
            print("   This may lead to disqualification!")
        else:
            print(f"✅ Success rate meets requirement ({min_valid}%)")
        
        # Check time constraints
        time_per_question = avg_time
        if time_per_question > 10:
            print(f"⚠️  WARNING: Average time per question ({time_per_question:.2f}s) exceeds 10s limit")
        else:
            print(f"✅ Time per question is within limits")
        
        # Save questions
        print(f"\n💾 Saving questions...")
        save_questions(valid_questions, args.output_file, args.verbose)
        
        # Final summary
        print(f"\n" + "="*50)
        print("GENERATION SUMMARY")
        print("="*50)
        print(f"Questions requested: {args.num_questions}")
        print(f"Questions generated: {len(questions)}")
        print(f"Valid questions: {len(valid_questions)}")
        print(f"Success rate: {valid_percentage:.1f}%")
        print(f"Total time: {generation_time:.2f}s")
        print(f"Avg time/question: {avg_time:.2f}s")
        print(f"Output file: {args.output_file}")
        
        if valid_percentage >= min_valid and time_per_question <= 10:
            print("🎉 All requirements met!")
            return 0
        else:
            print("⚠️  Some requirements not met - check warnings above")
            return 1
        
    except Exception as e:
        print(f"❌ Error during question generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        try:
            del question_model
            print("🧹 Model cleanup completed")
        except:
            pass

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)