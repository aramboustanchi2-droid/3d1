"""
Fine-Tuning Module for KURDO-AI
Enables custom training of external AI models on domain-specific data
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests

logger = logging.getLogger(__name__)

class FineTuningManager:
    """
    Manages fine-tuning operations for external AI models.
    Supports OpenAI, Anthropic (via prompt caching), HuggingFace, and local models.
    """
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "cad3d/super_ai/fine_tuning_config.json"
        self.jobs_history = []
        self.training_data_dir = "datasets/fine_tuning"
        
        # Load API keys from environment
        from dotenv import load_dotenv
        load_dotenv()
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.huggingface_token = os.getenv("HUGGINGFACE_API_KEY")
        
        # Ensure training data directory exists
        os.makedirs(self.training_data_dir, exist_ok=True)
        
        logger.info("Fine-Tuning Manager initialized")
    
    # ==========================================
    # OPENAI FINE-TUNING
    # ==========================================
    
    def create_openai_training_file(self, training_data: List[Dict], purpose: str = "fine-tune") -> Optional[str]:
        """
        Upload training data to OpenAI for fine-tuning.
        Format: [{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}]
        """
        if not self.openai_api_key:
            logger.error("OpenAI API key not found")
            return None
        
        try:
            # Save training data to JSONL file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            local_file = os.path.join(self.training_data_dir, f"openai_training_{timestamp}.jsonl")
            
            with open(local_file, 'w', encoding='utf-8') as f:
                for item in training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"Training data saved to {local_file}")
            
            # Upload to OpenAI
            url = "https://api.openai.com/v1/files"
            headers = {"Authorization": f"Bearer {self.openai_api_key}"}
            
            with open(local_file, 'rb') as f:
                files = {'file': f}
                data = {'purpose': purpose}
                response = requests.post(url, headers=headers, files=files, data=data)
            
            response.raise_for_status()
            file_info = response.json()
            file_id = file_info['id']
            
            logger.info(f"Training file uploaded to OpenAI: {file_id}")
            return file_id
        
        except Exception as e:
            logger.error(f"Failed to upload training file to OpenAI: {e}")
            return None
    
    def start_openai_fine_tune(
        self, 
        training_file_id: str, 
        model: str = "gpt-4o-mini-2024-07-18",
        suffix: Optional[str] = None,
        hyperparameters: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Start an OpenAI fine-tuning job.
        Supported base models: gpt-4o-mini-2024-07-18, gpt-3.5-turbo-0125
        """
        if not self.openai_api_key:
            logger.error("OpenAI API key not found")
            return None
        
        try:
            url = "https://api.openai.com/v1/fine_tuning/jobs"
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "training_file": training_file_id,
                "model": model
            }
            
            if suffix:
                data["suffix"] = suffix
            
            if hyperparameters:
                data["hyperparameters"] = hyperparameters
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            job_info = response.json()
            job_id = job_info['id']
            
            # Save to history
            self.jobs_history.append({
                "provider": "openai",
                "job_id": job_id,
                "model": model,
                "status": "started",
                "started_at": datetime.now().isoformat()
            })
            
            logger.info(f"OpenAI fine-tuning job started: {job_id}")
            return job_id
        
        except Exception as e:
            logger.error(f"Failed to start OpenAI fine-tuning: {e}")
            return None
    
    def check_openai_fine_tune_status(self, job_id: str) -> Optional[Dict]:
        """Check the status of an OpenAI fine-tuning job."""
        if not self.openai_api_key:
            return None
        
        try:
            url = f"https://api.openai.com/v1/fine_tuning/jobs/{job_id}"
            headers = {"Authorization": f"Bearer {self.openai_api_key}"}
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            return response.json()
        
        except Exception as e:
            logger.error(f"Failed to check OpenAI fine-tuning status: {e}")
            return None
    
    def list_openai_fine_tune_jobs(self, limit: int = 10) -> List[Dict]:
        """List all OpenAI fine-tuning jobs."""
        if not self.openai_api_key:
            return []
        
        try:
            url = f"https://api.openai.com/v1/fine_tuning/jobs?limit={limit}"
            headers = {"Authorization": f"Bearer {self.openai_api_key}"}
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            return response.json().get('data', [])
        
        except Exception as e:
            logger.error(f"Failed to list OpenAI fine-tuning jobs: {e}")
            return []
    
    # ==========================================
    # HUGGINGFACE FINE-TUNING
    # ==========================================
    
    def prepare_huggingface_dataset(
        self, 
        training_data: List[Dict],
        dataset_name: str = "kurdo_ai_training"
    ) -> str:
        """
        Prepare training data in HuggingFace format and save locally.
        Returns path to the saved dataset.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir = os.path.join(self.training_data_dir, f"{dataset_name}_{timestamp}")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save as JSON
        train_file = os.path.join(dataset_dir, "train.json")
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"HuggingFace dataset prepared at {dataset_dir}")
        return dataset_dir
    
    def start_huggingface_fine_tune(
        self,
        dataset_path: str,
        model_name: str = "google/flan-t5-base",
        output_dir: str = "models/fine_tuned"
    ) -> Dict:
        """
        Start fine-tuning a HuggingFace model locally.
        Note: This requires transformers and datasets libraries.
        """
        try:
            # Import required libraries
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
            from datasets import load_dataset
            
            logger.info(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Load dataset
            dataset = load_dataset('json', data_files={'train': os.path.join(dataset_path, 'train.json')})
            
            # Tokenization function
            def tokenize_function(examples):
                if "input" in examples and "output" in examples:
                    inputs = tokenizer(examples["input"], padding="max_length", truncation=True, max_length=512)
                    targets = tokenizer(examples["output"], padding="max_length", truncation=True, max_length=512)
                    inputs["labels"] = targets["input_ids"]
                    return inputs
                return examples
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="no",
                learning_rate=2e-5,
                per_device_train_batch_size=4,
                num_train_epochs=3,
                weight_decay=0.01,
                save_strategy="epoch",
                logging_dir='./logs',
                logging_steps=10,
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                tokenizer=tokenizer,
            )
            
            # Start training
            logger.info("Starting HuggingFace fine-tuning...")
            trainer.train()
            
            # Save model
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Model fine-tuned and saved to {output_dir}")
            
            return {
                "status": "completed",
                "output_dir": output_dir,
                "model_name": model_name
            }
        
        except ImportError:
            logger.error("transformers or datasets library not installed. Install with: pip install transformers datasets")
            return {"status": "error", "message": "Required libraries not installed"}
        
        except Exception as e:
            logger.error(f"HuggingFace fine-tuning failed: {e}")
            return {"status": "error", "message": str(e)}
    
    # ==========================================
    # ANTHROPIC PROMPT CACHING
    # ==========================================
    
    def create_anthropic_system_prompt_cache(
        self,
        system_prompt: str,
        training_examples: List[Dict]
    ) -> str:
        """
        Create a cached system prompt for Anthropic Claude with training examples.
        This simulates fine-tuning by including examples in the system prompt.
        """
        cached_prompt = f"{system_prompt}\n\n"
        cached_prompt += "# Training Examples (Few-Shot Learning)\n\n"
        
        for idx, example in enumerate(training_examples[:20]):  # Limit to 20 examples
            user_msg = example.get("input", example.get("user", ""))
            assistant_msg = example.get("output", example.get("assistant", ""))
            
            cached_prompt += f"Example {idx + 1}:\n"
            cached_prompt += f"User: {user_msg}\n"
            cached_prompt += f"Assistant: {assistant_msg}\n\n"
        
        cached_prompt += "Now, please respond to the following request in a similar manner:\n"
        
        logger.info(f"Created Anthropic cached prompt with {len(training_examples)} examples")
        return cached_prompt
    
    # ==========================================
    # ARCHITECTURAL DOMAIN TRAINING
    # ==========================================
    
    def prepare_architectural_training_data(
        self,
        source_dir: str = "datasets/persian_corpus/architecture"
    ) -> List[Dict]:
        """
        Prepare training data from architectural domain corpus.
        Converts Persian architectural knowledge into training examples.
        """
        training_data = []
        
        try:
            # Load architectural terms
            terms_file = os.path.join(source_dir, "architectural_terms.txt")
            if os.path.exists(terms_file):
                with open(terms_file, 'r', encoding='utf-8') as f:
                    terms = [line.strip() for line in f if line.strip()]
                
                for term in terms:
                    training_data.append({
                        "messages": [
                            {"role": "system", "content": "You are KURDO-AI, an expert in architecture and engineering."},
                            {"role": "user", "content": f"Explain the architectural term: {term}"},
                            {"role": "assistant", "content": f"{term} is an important architectural concept used in building design and construction."}
                        ]
                    })
            
            # Load feasibility guidelines
            guidelines_file = os.path.join(source_dir, "feasibility_guidelines.txt")
            if os.path.exists(guidelines_file):
                with open(guidelines_file, 'r', encoding='utf-8') as f:
                    guidelines = f.read().split('\n\n')
                
                for guideline in guidelines[:50]:  # Limit to 50
                    if guideline.strip():
                        training_data.append({
                            "messages": [
                                {"role": "system", "content": "You are KURDO-AI, an expert in architectural feasibility analysis."},
                                {"role": "user", "content": "Provide a feasibility guideline for architectural projects"},
                                {"role": "assistant", "content": guideline.strip()}
                            ]
                        })
            
            logger.info(f"Prepared {len(training_data)} architectural training examples")
            return training_data
        
        except Exception as e:
            logger.error(f"Failed to prepare architectural training data: {e}")
            return []
    
    # ==========================================
    # WORKFLOW METHODS
    # ==========================================
    
    def full_fine_tune_workflow(
        self,
        provider: str = "openai",
        training_data: Optional[List[Dict]] = None,
        base_model: str = "gpt-4o-mini-2024-07-18",
        custom_suffix: str = "kurdo-ai-arch"
    ) -> Dict:
        """
        Complete fine-tuning workflow from data preparation to model training.
        
        Args:
            provider: "openai", "huggingface", or "anthropic"
            training_data: List of training examples
            base_model: Base model to fine-tune
            custom_suffix: Suffix for the fine-tuned model name
        """
        result = {
            "status": "started",
            "provider": provider,
            "timestamp": datetime.now().isoformat()
        }
        
        # Use architectural data if none provided
        if not training_data:
            logger.info("No training data provided, using architectural corpus...")
            training_data = self.prepare_architectural_training_data()
            
            if not training_data:
                result["status"] = "error"
                result["message"] = "No training data available"
                return result
        
        try:
            if provider == "openai":
                # OpenAI workflow
                logger.info("Starting OpenAI fine-tuning workflow...")
                
                # Upload training file
                file_id = self.create_openai_training_file(training_data)
                if not file_id:
                    result["status"] = "error"
                    result["message"] = "Failed to upload training file"
                    return result
                
                result["training_file_id"] = file_id
                
                # Start fine-tuning job
                job_id = self.start_openai_fine_tune(
                    training_file_id=file_id,
                    model=base_model,
                    suffix=custom_suffix
                )
                
                if not job_id:
                    result["status"] = "error"
                    result["message"] = "Failed to start fine-tuning job"
                    return result
                
                result["job_id"] = job_id
                result["status"] = "running"
                result["message"] = f"Fine-tuning job started. Check status with job_id: {job_id}"
            
            elif provider == "huggingface":
                # HuggingFace workflow
                logger.info("Starting HuggingFace fine-tuning workflow...")
                
                # Prepare dataset
                dataset_path = self.prepare_huggingface_dataset(training_data)
                result["dataset_path"] = dataset_path
                
                # Start fine-tuning
                training_result = self.start_huggingface_fine_tune(
                    dataset_path=dataset_path,
                    model_name=base_model
                )
                
                result.update(training_result)
            
            elif provider == "anthropic":
                # Anthropic prompt caching (simulated fine-tuning)
                logger.info("Creating Anthropic cached prompt with training examples...")
                
                system_prompt = "You are KURDO-AI, an advanced architectural and engineering AI system with expertise in feasibility analysis, design optimization, and construction planning."
                
                cached_prompt = self.create_anthropic_system_prompt_cache(
                    system_prompt=system_prompt,
                    training_examples=training_data
                )
                
                # Save cached prompt
                cache_file = os.path.join(self.training_data_dir, f"anthropic_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(cached_prompt)
                
                result["status"] = "completed"
                result["cached_prompt_file"] = cache_file
                result["message"] = "Anthropic cached prompt created with training examples"
            
            else:
                result["status"] = "error"
                result["message"] = f"Unsupported provider: {provider}"
            
            # Save workflow result
            self._save_workflow_result(result)
            
            return result
        
        except Exception as e:
            logger.error(f"Fine-tuning workflow failed: {e}")
            result["status"] = "error"
            result["message"] = str(e)
            return result
    
    def _save_workflow_result(self, result: Dict):
        """Save fine-tuning workflow result to history file."""
        history_file = os.path.join(self.training_data_dir, "fine_tuning_history.json")
        
        try:
            # Load existing history
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = []
            
            # Append new result
            history.append(result)
            
            # Save
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Workflow result saved to {history_file}")
        
        except Exception as e:
            logger.error(f"Failed to save workflow result: {e}")
    
    def get_fine_tuning_history(self) -> List[Dict]:
        """Get history of all fine-tuning operations."""
        history_file = os.path.join(self.training_data_dir, "fine_tuning_history.json")
        
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return []


# Global instance
fine_tuning_manager = FineTuningManager()
