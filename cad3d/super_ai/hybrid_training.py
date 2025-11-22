"""
Hybrid Training Manager for KURDO-AI
Intelligently chooses between Full Fine-Tuning and LoRA based on:
- Available resources (GPU memory, compute)
- Dataset size
- Model size
- Time constraints
- Budget
"""

import logging
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from .fine_tuning import fine_tuning_manager
    FINE_TUNING_AVAILABLE = True
except ImportError:
    FINE_TUNING_AVAILABLE = False

try:
    from .lora_training import lora_manager
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False

try:
    from .prompt_engineering import prompt_engineering_manager
    PROMPT_ENGINEERING_AVAILABLE = True
except ImportError:
    PROMPT_ENGINEERING_AVAILABLE = False

try:
    from .rag_system import rag_system
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


class TrainingMethod(Enum):
    """Training method selection."""
    FULL_FINETUNING = "full_finetuning"
    LORA = "lora"
    LORA_4BIT = "lora_4bit"  # LoRA with 4-bit quantization
    LORA_8BIT = "lora_8bit"  # LoRA with 8-bit quantization
    PROMPT_CACHING = "prompt_caching"  # Anthropic style
    PROMPT_ENGINEERING = "prompt_engineering"  # Zero/Few-shot learning
    RAG = "rag"  # Retrieval-Augmented Generation


class HybridTrainingManager:
    """
    Intelligent training manager that selects the best method automatically.
    """
    
    def __init__(self):
        self.methods_available = {
            "full_finetuning": FINE_TUNING_AVAILABLE,
            "lora": LORA_AVAILABLE,
            "prompt_caching": FINE_TUNING_AVAILABLE,
            "prompt_engineering": PROMPT_ENGINEERING_AVAILABLE,
            "rag": RAG_AVAILABLE
        }
        logger.info(f"Hybrid Training Manager initialized. Available methods: {[k for k, v in self.methods_available.items() if v]}")
    
    def recommend_method(
        self,
        model_size_gb: float = 7.0,
        dataset_size: int = 100,
        gpu_memory_gb: Optional[float] = None,
        training_time_hours: Optional[float] = None,
        budget_usd: Optional[float] = None,
        provider: str = "local"
    ) -> Dict[str, Any]:
        """
        Recommend the best training method based on constraints.
        
        Args:
            model_size_gb: Model size in GB (e.g., 7 for Llama-2-7B)
            dataset_size: Number of training samples
            gpu_memory_gb: Available GPU memory (None = auto-detect or assume CPU)
            training_time_hours: Maximum time budget
            budget_usd: Budget for cloud training (None = local training)
            provider: Target provider ("openai", "huggingface", "anthropic", "local")
        
        Returns:
            Recommendation with method, reasoning, and estimated costs
        """
        recommendations = []
        
        # Decision tree for method selection
        
        # 1. Check if cloud API is preferred
        if provider == "openai":
            if budget_usd and budget_usd >= 10:
                recommendations.append({
                    "method": TrainingMethod.FULL_FINETUNING,
                    "provider": "openai",
                    "confidence": 0.9,
                    "reasoning": [
                        "OpenAI provides managed fine-tuning",
                        "No local GPU required",
                        "Fast turnaround (30-60 min)",
                        f"Budget ${budget_usd} sufficient"
                    ],
                    "estimated_cost_usd": 5 + (dataset_size * 0.0001),
                    "estimated_time_hours": 0.5,
                    "requirements": ["OpenAI API key", f"Budget: ~$5-10"]
                })
            else:
                recommendations.append({
                    "method": "prompt_caching",
                    "provider": "openai",
                    "confidence": 0.7,
                    "reasoning": [
                        "Budget too low for fine-tuning",
                        "Use few-shot learning instead",
                        "No training required"
                    ],
                    "estimated_cost_usd": 0.1 * dataset_size,
                    "estimated_time_hours": 0,
                    "requirements": ["OpenAI API key"]
                })
        
        elif provider == "anthropic":
            recommendations.append({
                "method": "prompt_caching",
                "provider": "anthropic",
                "confidence": 0.8,
                "reasoning": [
                    "Anthropic doesn't offer fine-tuning",
                    "Use cached prompts with examples",
                    "Cost-effective for repeated use"
                ],
                "estimated_cost_usd": 0.05 * dataset_size,
                "estimated_time_hours": 0,
                "requirements": ["Anthropic API key"]
            })
        
        # 2. Local/HuggingFace training
        elif provider in ["local", "huggingface"]:
            # Detect GPU if not provided
            if gpu_memory_gb is None:
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    else:
                        gpu_memory_gb = 0
                except:
                    gpu_memory_gb = 0
            
            # Decision based on available resources
            
            # Option 1: Full fine-tuning (requires powerful GPU)
            required_memory_full = model_size_gb * 4  # Rough estimate
            if gpu_memory_gb >= required_memory_full:
                recommendations.append({
                    "method": TrainingMethod.FULL_FINETUNING,
                    "provider": "huggingface",
                    "confidence": 0.8,
                    "reasoning": [
                        f"GPU memory ({gpu_memory_gb:.1f}GB) sufficient",
                        "Full fine-tuning gives best quality",
                        "Complete model adaptation"
                    ],
                    "estimated_cost_usd": 0,  # Local training
                    "estimated_time_hours": 2 + (dataset_size / 50),
                    "requirements": [
                        f"GPU: {required_memory_full:.0f}GB+ VRAM",
                        "transformers, datasets, torch"
                    ]
                })
            
            # Option 2: LoRA with 8-bit (moderate GPU)
            required_memory_lora8 = model_size_gb * 0.5
            if gpu_memory_gb >= required_memory_lora8:
                recommendations.append({
                    "method": TrainingMethod.LORA_8BIT,
                    "provider": "huggingface",
                    "confidence": 0.95,
                    "reasoning": [
                        f"LoRA uses 90% fewer parameters",
                        "8-bit quantization fits in GPU",
                        "3-10x faster than full fine-tuning",
                        "Quality nearly as good as full FT"
                    ],
                    "estimated_cost_usd": 0,
                    "estimated_time_hours": 0.5 + (dataset_size / 100),
                    "requirements": [
                        f"GPU: {required_memory_lora8:.0f}GB+ VRAM (e.g., RTX 3060)",
                        "peft, transformers, bitsandbytes"
                    ]
                })
            
            # Option 3: LoRA with 4-bit (works on most GPUs)
            required_memory_lora4 = model_size_gb * 0.25
            if gpu_memory_gb >= required_memory_lora4 or gpu_memory_gb >= 6:
                recommendations.append({
                    "method": TrainingMethod.LORA_4BIT,
                    "provider": "huggingface",
                    "confidence": 0.9,
                    "reasoning": [
                        "4-bit quantization works on consumer GPUs",
                        "LoRA is extremely parameter-efficient",
                        "Can train 7B model on 6GB GPU",
                        "Good quality with minimal resources"
                    ],
                    "estimated_cost_usd": 0,
                    "estimated_time_hours": 1 + (dataset_size / 80),
                    "requirements": [
                        "GPU: 6GB+ VRAM (e.g., RTX 2060)",
                        "peft, transformers, bitsandbytes"
                    ]
                })
            
            # Option 4: Prompt Engineering (no GPU needed)
            if PROMPT_ENGINEERING_AVAILABLE:
                recommendations.append({
                    "method": TrainingMethod.PROMPT_ENGINEERING,
                    "provider": "local",
                    "confidence": 0.85 if gpu_memory_gb < 6 else 0.7,
                    "reasoning": [
                        "No training required (zero-shot/few-shot)",
                        "Works with any GPU or CPU",
                        "Instant setup - immediate results",
                        "Cost-effective for prototyping",
                        "Great for small datasets or changing tasks"
                    ],
                    "estimated_cost_usd": 0,
                    "estimated_time_hours": 0,
                    "requirements": ["Any inference API (OpenAI, Anthropic, local models)"]
                })
            
            # Option 5: CPU-based (last resort)
            if gpu_memory_gb < 6 or gpu_memory_gb == 0:
                recommendations.append({
                    "method": "prompt_caching",
                    "provider": "local",
                    "confidence": 0.5,
                    "reasoning": [
                        "Insufficient GPU memory for training",
                        "Consider cloud APIs (OpenAI/Anthropic)",
                        "Or use prompt engineering instead"
                    ],
                    "estimated_cost_usd": 0,
                    "estimated_time_hours": float('inf'),
                    "requirements": ["Consider using cloud APIs or prompt engineering"]
                })
        
        # 3. Apply time and budget constraints
        if training_time_hours:
            recommendations = [r for r in recommendations if r["estimated_time_hours"] <= training_time_hours]
        
        if budget_usd:
            recommendations = [r for r in recommendations if r["estimated_cost_usd"] <= budget_usd]
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        
        if not recommendations:
            return {
                "recommended_method": None,
                "message": "No suitable method found with given constraints",
                "suggestions": [
                    "Increase GPU memory (get better GPU or use cloud)",
                    "Increase budget for cloud fine-tuning",
                    "Use prompt caching/few-shot learning instead"
                ]
            }
        
        best = recommendations[0]
        
        return {
            "recommended_method": best["method"],
            "provider": best["provider"],
            "confidence": best["confidence"],
            "reasoning": best["reasoning"],
            "estimated_cost_usd": best["estimated_cost_usd"],
            "estimated_time_hours": best["estimated_time_hours"],
            "requirements": best["requirements"],
            "alternatives": recommendations[1:3] if len(recommendations) > 1 else [],
            "gpu_memory_available": gpu_memory_gb,
            "dataset_size": dataset_size
        }
    
    def auto_train(
        self,
        training_data: List[Dict],
        adapter_name: str = "kurdo-ai-auto",
        model_name: str = "meta-llama/Llama-2-7b-hf",
        provider: str = "local",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Automatically select and execute the best training method.
        
        Args:
            training_data: Training examples
            adapter_name: Name for the adapter/model
            model_name: Base model to use
            provider: Preferred provider
        
        Returns:
            Training results
        """
        # Get model size (rough estimate)
        model_size_map = {
            "llama-2-7b": 7.0,
            "llama-2-13b": 13.0,
            "flan-t5-base": 0.9,
            "flan-t5-large": 3.0,
            "gpt-3.5": 20.0,  # Unknown exact size
            "gpt-4": 1000.0   # Very large
        }
        
        model_size = 7.0  # Default
        for key, size in model_size_map.items():
            if key in model_name.lower():
                model_size = size
                break
        
        # Get recommendation
        recommendation = self.recommend_method(
            model_size_gb=model_size,
            dataset_size=len(training_data),
            provider=provider,
            **kwargs
        )
        
        logger.info(f"Auto-training recommendation: {recommendation['recommended_method']}")
        logger.info(f"Reasoning: {recommendation.get('reasoning', [])}")
        
        method = recommendation.get("recommended_method")
        
        if not method:
            return {
                "status": "error",
                "message": "No suitable training method found",
                "recommendation": recommendation
            }
        
        # Execute training based on selected method
        if method == TrainingMethod.FULL_FINETUNING:
            if provider == "openai" and FINE_TUNING_AVAILABLE:
                return fine_tuning_manager.full_fine_tune_workflow(
                    provider="openai",
                    training_data=training_data,
                    custom_suffix=adapter_name
                )
            elif provider in ["huggingface", "local"] and FINE_TUNING_AVAILABLE:
                return fine_tuning_manager.full_fine_tune_workflow(
                    provider="huggingface",
                    training_data=training_data,
                    base_model=model_name
                )
        
        elif method in [TrainingMethod.LORA, TrainingMethod.LORA_8BIT, TrainingMethod.LORA_4BIT]:
            if not LORA_AVAILABLE:
                return {
                    "status": "error",
                    "message": "LoRA not available. Install with: pip install peft bitsandbytes"
                }
            
            load_in_8bit = method == TrainingMethod.LORA_8BIT
            load_in_4bit = method == TrainingMethod.LORA_4BIT
            
            # Convert data format if needed
            lora_data = []
            for item in training_data:
                if "messages" in item:
                    msgs = item["messages"]
                    user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                    assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                    if user_msg and assistant_msg:
                        lora_data.append({"prompt": user_msg, "completion": assistant_msg})
                elif "prompt" in item and "completion" in item:
                    lora_data.append(item)
                elif "input" in item and "output" in item:
                    lora_data.append({"prompt": item["input"], "completion": item["output"]})
            
            return lora_manager.train_lora_adapter(
                model_name=model_name,
                training_data=lora_data,
                adapter_name=adapter_name,
                load_in_8bit=load_in_8bit or False,
                **kwargs
            )
        
        elif method == TrainingMethod.PROMPT_ENGINEERING:
            if not PROMPT_ENGINEERING_AVAILABLE:
                return {
                    "status": "error",
                    "message": "Prompt Engineering module not available"
                }
            
            # Convert data to examples
            examples = []
            for item in training_data:
                if "messages" in item:
                    msgs = item["messages"]
                    user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                    assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                    if user_msg and assistant_msg:
                        examples.append({"input": user_msg, "output": assistant_msg})
                elif "prompt" in item and "completion" in item:
                    examples.append({"input": item["prompt"], "output": item["completion"]})
                elif "input" in item and "output" in item:
                    examples.append(item)
            
            # Create few-shot prompt or cached system prompt
            cached_prompt = prompt_engineering_manager.create_cached_system_prompt(
                system_role=f"You are KURDO-AI, trained as {adapter_name}",
                training_examples=examples
            )
            
            # Also save as instruction set
            from .prompt_engineering import InstructionSet
            instruction_set = InstructionSet(
                name=adapter_name,
                instructions=[
                    "Follow the patterns from the examples",
                    "Maintain consistent style and format",
                    "Provide accurate, professional responses"
                ],
                examples=examples[:20]  # Store top 20 examples
            )
            prompt_engineering_manager.add_instruction_set(instruction_set)
            
            return {
                "status": "success",
                "method": "prompt_engineering",
                "adapter_name": adapter_name,
                "cached_prompt": cached_prompt,
                "instruction_set": instruction_set.name,
                "num_examples": len(examples),
                "message": "Prompt engineering setup complete. Use cached prompt or instruction set for inference.",
                "usage": {
                    "cached_prompt": "Use as system message in API calls",
                    "instruction_set": f"prompt_engineering_manager.instruction_sets['{instruction_set.name}']"
                }
            }
        
        elif method == "prompt_caching":
            if provider == "anthropic" and FINE_TUNING_AVAILABLE:
                # Convert data
                examples = []
                for item in training_data:
                    if "messages" in item:
                        msgs = item["messages"]
                        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                        assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                        examples.append({"input": user_msg, "output": assistant_msg})
                    elif "prompt" in item:
                        examples.append({"input": item["prompt"], "output": item["completion"]})
                
                cached_prompt = fine_tuning_manager.create_anthropic_system_prompt_cache(
                    system_prompt="You are KURDO-AI, expert in architecture and engineering.",
                    training_examples=examples
                )
                
                return {
                    "status": "success",
                    "method": "prompt_caching",
                    "cached_prompt": cached_prompt,
                    "message": "Use this cached prompt with Anthropic API for best results"
                }
        
        return {
            "status": "error",
            "message": f"Method {method} not implemented yet",
            "recommendation": recommendation
        }
    
    def compare_all_methods(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        dataset_size: int = 100
    ) -> Dict[str, Any]:
        """
        Generate comprehensive comparison of all training methods.
        
        Returns:
            Detailed comparison table
        """
        comparison = {
            "model": model_name,
            "dataset_size": dataset_size,
            "methods": {}
        }
        
        # Prompt Engineering
        comparison["methods"]["Prompt Engineering"] = {
            "pros": [
                "Zero training time (instant setup)",
                "Works with any model/API",
                "No GPU required",
                "Extremely flexible",
                "Great for prototyping",
                "Can use pre-trained models"
            ],
            "cons": [
                "Token limits (context window)",
                "Less consistent than fine-tuning",
                "Repetitive examples increase cost",
                "Requires prompt engineering skills"
            ],
            "best_for": [
                "No training data",
                "Rapid prototyping",
                "Changing requirements",
                "Zero-shot tasks",
                "Cost-sensitive applications"
            ],
            "estimated_time": "0 (instant)",
            "estimated_cost": "$0 (+ inference costs)",
            "gpu_required": "None"
        }
        
        # Full Fine-Tuning
        comparison["methods"]["Full Fine-Tuning"] = {
            "pros": [
                "Best quality results",
                "Complete model adaptation",
                "No architecture changes needed"
            ],
            "cons": [
                "Requires powerful GPU (40GB+ VRAM)",
                "Slow training (hours to days)",
                "Expensive ($10-100+ for cloud)",
                "Large model files (~14GB)"
            ],
            "best_for": [
                "Maximum quality requirements",
                "Large budgets",
                "Access to powerful GPUs"
            ],
            "estimated_time": "2-10 hours",
            "estimated_cost": "$10-50 (cloud) or $0 (local with GPU)",
            "gpu_required": "A100 40GB or better"
        }
        
        # LoRA (8-bit)
        comparison["methods"]["LoRA (8-bit)"] = {
            "pros": [
                "90-99% fewer parameters",
                "3-10x faster than full FT",
                "Works on consumer GPUs (12GB+)",
                "Small adapter files (~50MB)",
                "Nearly same quality as full FT"
            ],
            "cons": [
                "Still needs decent GPU",
                "Slightly lower quality than full FT",
                "More complex setup"
            ],
            "best_for": [
                "Most users (best balance)",
                "Consumer GPUs",
                "Multiple task-specific adapters"
            ],
            "estimated_time": "0.5-2 hours",
            "estimated_cost": "$0 (local)",
            "gpu_required": "RTX 3060 12GB or better"
        }
        
        # LoRA (4-bit)
        comparison["methods"]["LoRA (4-bit)"] = {
            "pros": [
                "Works on 6GB GPUs!",
                "Very fast training",
                "Extremely low memory",
                "Good quality"
            ],
            "cons": [
                "Lower quality than 8-bit",
                "Some precision loss",
                "Not all models supported"
            ],
            "best_for": [
                "Limited GPU memory",
                "Quick experiments",
                "Budget constraints"
            ],
            "estimated_time": "1-3 hours",
            "estimated_cost": "$0 (local)",
            "gpu_required": "RTX 2060 6GB or better"
        }
        
        # OpenAI Fine-Tuning
        comparison["methods"]["OpenAI Fine-Tuning"] = {
            "pros": [
                "No local GPU needed",
                "Managed service (easy)",
                "Fast (30-60 min)",
                "Good quality"
            ],
            "cons": [
                "Costs money ($5-20 per job)",
                "Data sent to OpenAI",
                "Less control",
                "Limited to GPT-3.5/4o-mini"
            ],
            "best_for": [
                "No local GPU",
                "Quick turnaround",
                "Production deployments"
            ],
            "estimated_time": "0.5-1 hour",
            "estimated_cost": "$5-20",
            "gpu_required": "None (cloud)"
        }
        
        # Prompt Caching
        comparison["methods"]["Prompt Caching (Anthropic)"] = {
            "pros": [
                "No training needed",
                "Instant setup",
                "Very cheap",
                "No GPU required"
            ],
            "cons": [
                "Not real fine-tuning",
                "Limited effectiveness",
                "Must send examples each time",
                "Only works with Claude"
            ],
            "best_for": [
                "Quick prototypes",
                "Small datasets",
                "No training budget"
            ],
            "estimated_time": "0 (instant)",
            "estimated_cost": "$0.1-1 per session",
            "gpu_required": "None"
        }
        
        # RAG (Retrieval-Augmented Generation)
        comparison["methods"]["RAG"] = {
            "pros": [
                "No training required (instant)",
                "Easily updatable (add new docs)",
                "Transparent (shows sources)",
                "Works with any model",
                "Excellent for knowledge-based tasks",
                "No model drift"
            ],
            "cons": [
                "Depends on retrieval quality",
                "Context window limits",
                "Requires good documents",
                "May retrieve irrelevant info"
            ],
            "best_for": [
                "Large knowledge bases",
                "Frequently updated information",
                "Fact-based queries",
                "Domain-specific knowledge",
                "Multi-document reasoning"
            ],
            "estimated_time": "Minutes (indexing)",
            "estimated_cost": "$0 (+ inference)",
            "gpu_required": "None (optional for embeddings)"
        }
        
        # Add recommendation
        rec = self.recommend_method(
            model_size_gb=7.0,
            dataset_size=dataset_size,
            provider="local"
        )
        
        comparison["recommendation"] = rec
        
        return comparison


# Global instance
hybrid_manager = HybridTrainingManager()
