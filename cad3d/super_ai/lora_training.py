"""
LoRA (Low-Rank Adaptation) Module for KURDO-AI
Efficient fine-tuning using parameter-efficient methods.
LoRA adds trainable low-rank matrices to model layers, reducing training cost by 90%+.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from peft import prepare_model_for_kbit_training
    import transformers
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT library not available. Install with: pip install peft")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class LoRAManager:
    """
    Manages LoRA (Low-Rank Adaptation) training for efficient fine-tuning.
    
    LoRA advantages:
    - 90-99% fewer trainable parameters
    - 3-10x faster training
    - Much lower memory requirements
    - Easy to switch between adapters
    - Can train on consumer GPUs (RTX 3060+)
    """
    
    def __init__(self):
        self.lora_dir = Path("models/lora_adapters")
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_history = []
        self.history_file = self.lora_dir / "training_history.json"
        self._load_history()
        
        logger.info("LoRA Manager initialized")
    
    def _load_history(self):
        """Load training history from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.training_history = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load history: {e}")
                self.training_history = []
    
    def _save_history(self):
        """Save training history to disk."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def create_lora_config(
        self,
        task_type: str = "CAUSAL_LM",
        r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        **kwargs
    ) -> LoraConfig:
        """
        Create LoRA configuration.
        
        Args:
            task_type: Type of task (CAUSAL_LM, SEQ_2_SEQ_LM, etc.)
            r: Rank of LoRA matrices (lower = fewer parameters, typically 4-64)
            lora_alpha: Scaling factor (typically 2*r to 4*r)
            lora_dropout: Dropout probability
            target_modules: Which layers to adapt (None = auto-detect)
        
        Returns:
            LoraConfig object
        """
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library not installed. Install with: pip install peft")
        
        # Map string to TaskType enum
        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "QUESTION_ANS": TaskType.QUESTION_ANS,
            "FEATURE_EXTRACTION": TaskType.FEATURE_EXTRACTION
        }
        
        task = task_type_map.get(task_type, TaskType.CAUSAL_LM)
        
        # Default target modules for common architectures
        if target_modules is None:
            # Common patterns that work for most models
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        config = LoraConfig(
            task_type=task,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            inference_mode=False,
            **kwargs
        )
        
        logger.info(f"Created LoRA config: rank={r}, alpha={lora_alpha}, dropout={lora_dropout}")
        logger.info(f"Target modules: {target_modules}")
        
        return config
    
    def prepare_model_for_lora(
        self,
        model_name: str,
        lora_config: LoraConfig,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Load base model and add LoRA adapters.
        
        Args:
            model_name: HuggingFace model name or path
            lora_config: LoRA configuration
            load_in_8bit: Use 8-bit quantization (saves memory)
            load_in_4bit: Use 4-bit quantization (saves even more memory)
        
        Returns:
            tuple: (model, tokenizer)
        """
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library not installed")
        
        logger.info(f"Loading base model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with quantization if requested
        if load_in_8bit or load_in_4bit:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                device_map="auto",
                torch_dtype=torch.float16
            )
            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        # Add LoRA adapters
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / total_params
        
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_percent:.2f}%)")
        
        return model, tokenizer
    
    def train_lora_adapter(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        training_data: List[Dict] = None,
        adapter_name: str = "kurdo-ai-arch",
        r: int = 16,
        lora_alpha: int = 32,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        max_length: int = 512,
        load_in_8bit: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a LoRA adapter on custom data.
        
        Args:
            model_name: Base model to adapt
            training_data: List of training examples
            adapter_name: Name for the LoRA adapter
            r: LoRA rank (4-64, lower = fewer parameters)
            lora_alpha: LoRA alpha (scaling factor)
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            max_length: Maximum sequence length
            load_in_8bit: Use 8-bit quantization
        
        Returns:
            Training results and adapter path
        """
        if not PEFT_AVAILABLE:
            return {
                "status": "error",
                "message": "PEFT library not installed. Install with: pip install peft transformers accelerate bitsandbytes"
            }
        
        if not training_data:
            return {
                "status": "error",
                "message": "No training data provided"
            }
        
        try:
            logger.info(f"Starting LoRA training for {model_name}")
            logger.info(f"Training samples: {len(training_data)}")
            
            # Create LoRA config
            lora_config = self.create_lora_config(
                task_type="CAUSAL_LM",
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=0.1
            )
            
            # Load model with LoRA
            model, tokenizer = self.prepare_model_for_lora(
                model_name=model_name,
                lora_config=lora_config,
                load_in_8bit=load_in_8bit
            )
            
            # Prepare dataset
            def tokenize_function(examples):
                # Format: "Question: {prompt}\nAnswer: {completion}"
                texts = []
                for item in examples:
                    prompt = item.get("prompt", item.get("input", ""))
                    completion = item.get("completion", item.get("output", ""))
                    text = f"Question: {prompt}\nAnswer: {completion}"
                    texts.append(text)
                
                return tokenizer(
                    texts,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length"
                )
            
            # Tokenize training data
            logger.info("Tokenizing training data...")
            tokenized_data = [tokenize_function([item]) for item in training_data]
            
            # Prepare for DataLoader
            from torch.utils.data import Dataset, DataLoader
            
            class SimpleDataset(Dataset):
                def __init__(self, data):
                    self.data = data
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    item = self.data[idx]
                    return {
                        "input_ids": torch.tensor(item["input_ids"][0]),
                        "attention_mask": torch.tensor(item["attention_mask"][0]),
                        "labels": torch.tensor(item["input_ids"][0])
                    }
            
            train_dataset = SimpleDataset(tokenized_data)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Training setup
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            model.train()
            
            logger.info("Starting training...")
            total_steps = len(train_loader) * num_epochs
            step = 0
            
            for epoch in range(num_epochs):
                epoch_loss = 0
                for batch_idx, batch in enumerate(train_loader):
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    epoch_loss += loss.item()
                    step += 1
                    
                    if (batch_idx + 1) % 10 == 0:
                        logger.info(f"Epoch {epoch+1}/{num_epochs}, Step {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
                avg_loss = epoch_loss / len(train_loader)
                logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Save LoRA adapter
            adapter_path = self.lora_dir / adapter_name
            adapter_path.mkdir(exist_ok=True)
            
            model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)
            
            logger.info(f"LoRA adapter saved to {adapter_path}")
            
            # Save training record
            record = {
                "adapter_name": adapter_name,
                "base_model": model_name,
                "r": r,
                "lora_alpha": lora_alpha,
                "training_samples": len(training_data),
                "num_epochs": num_epochs,
                "final_loss": avg_loss,
                "adapter_path": str(adapter_path),
                "timestamp": datetime.now().isoformat()
            }
            
            self.training_history.append(record)
            self._save_history()
            
            return {
                "status": "success",
                "adapter_name": adapter_name,
                "adapter_path": str(adapter_path),
                "base_model": model_name,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "reduction_percent": 100 - trainable_percent,
                "final_loss": avg_loss,
                "record": record
            }
        
        except Exception as e:
            logger.error(f"LoRA training failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def load_lora_adapter(
        self,
        base_model: str,
        adapter_path: str,
        load_in_8bit: bool = True
    ):
        """
        Load a trained LoRA adapter.
        
        Args:
            base_model: Name of base model
            adapter_path: Path to LoRA adapter
            load_in_8bit: Use 8-bit quantization
        
        Returns:
            tuple: (model, tokenizer)
        """
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library not installed")
        
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        # Load base model
        if load_in_8bit:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        
        logger.info("LoRA adapter loaded successfully")
        return model, tokenizer
    
    def generate_with_lora(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generate text using LoRA-adapted model.
        
        Args:
            model: LoRA model
            tokenizer: Tokenizer
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated text
        """
        # Format prompt
        formatted_prompt = f"Question: {prompt}\nAnswer:"
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                **kwargs
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer part
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:", 1)[1].strip()
        else:
            answer = generated_text
        
        return answer
    
    def list_adapters(self) -> List[Dict[str, Any]]:
        """List all trained LoRA adapters."""
        adapters = []
        
        for adapter_dir in self.lora_dir.iterdir():
            if adapter_dir.is_dir() and (adapter_dir / "adapter_config.json").exists():
                try:
                    with open(adapter_dir / "adapter_config.json", 'r') as f:
                        config = json.load(f)
                    
                    adapters.append({
                        "name": adapter_dir.name,
                        "path": str(adapter_dir),
                        "config": config
                    })
                except Exception as e:
                    logger.warning(f"Failed to read adapter {adapter_dir}: {e}")
        
        return adapters
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get history of all LoRA training sessions."""
        return self.training_history
    
    def compare_with_full_finetuning(self, model_name: str, r: int = 16) -> Dict[str, Any]:
        """
        Compare LoRA vs full fine-tuning for a given model.
        
        Returns:
            Comparison metrics (parameters, memory, estimated time)
        """
        try:
            # Load model info (without loading weights)
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            
            # Estimate full model parameters
            # This is approximate, actual may vary
            total_params = config.num_parameters if hasattr(config, 'num_parameters') else 7_000_000_000
            
            # LoRA parameters estimation
            # Typically LoRA adds: (r * 2 * d) parameters per adapted layer
            # For a 7B model with 32 layers adapting 4 matrices each: 
            # ~16M parameters for r=8, ~32M for r=16
            lora_params = r * 2 * 4096 * 32 * 4  # Rough estimate
            
            reduction = 100 * (1 - lora_params / total_params)
            
            # Memory estimates (rough)
            full_ft_memory_gb = (total_params * 4) / (1024**3) * 4  # 4 bytes per param, 4x for gradients etc
            lora_memory_gb = (total_params * 2 + lora_params * 4) / (1024**3)  # Base in int8 + LoRA in fp32
            
            # Time estimates (very rough)
            full_ft_hours = 10  # Typical for 7B model
            lora_hours = full_ft_hours * (lora_params / total_params)
            
            return {
                "model": model_name,
                "full_finetuning": {
                    "trainable_params": total_params,
                    "memory_required_gb": round(full_ft_memory_gb, 1),
                    "estimated_time_hours": full_ft_hours,
                    "gpu_required": "A100 40GB or better"
                },
                "lora_r" + str(r): {
                    "trainable_params": lora_params,
                    "memory_required_gb": round(lora_memory_gb, 1),
                    "estimated_time_hours": round(lora_hours, 1),
                    "gpu_required": "RTX 3060 12GB or better",
                    "parameter_reduction": f"{reduction:.1f}%"
                },
                "advantages": [
                    f"LoRA uses {reduction:.0f}% fewer parameters",
                    f"LoRA uses ~{full_ft_memory_gb/lora_memory_gb:.1f}x less memory",
                    f"LoRA trains ~{full_ft_hours/lora_hours:.1f}x faster",
                    "LoRA adapters are small (~50MB vs ~14GB)",
                    "Can train multiple task-specific adapters",
                    "Easy to switch between adapters"
                ]
            }
        
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return {"error": str(e)}
    
    def merge_lora_to_base(
        self,
        base_model: str,
        adapter_path: str,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Merge LoRA adapter into base model (creates a standalone model).
        Useful for deployment when you don't need to switch adapters.
        
        Args:
            base_model: Base model name
            adapter_path: Path to LoRA adapter
            output_path: Where to save merged model
        
        Returns:
            Merge results
        """
        if not PEFT_AVAILABLE:
            return {"status": "error", "message": "PEFT library not installed"}
        
        try:
            logger.info("Loading base model and adapter...")
            
            # Load model with adapter
            model, tokenizer = self.load_lora_adapter(
                base_model=base_model,
                adapter_path=adapter_path,
                load_in_8bit=False  # Need full precision for merge
            )
            
            # Merge adapter weights into base model
            logger.info("Merging LoRA weights into base model...")
            model = model.merge_and_unload()
            
            # Save merged model
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            logger.info(f"Merged model saved to {output_path}")
            
            return {
                "status": "success",
                "merged_model_path": str(output_path),
                "message": "LoRA adapter merged into base model successfully"
            }
        
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            return {"status": "error", "message": str(e)}


# Global instance
lora_manager = LoRAManager()
