"""
Prompt Engineering & Instruction Tuning Module for KURDO-AI
Third complementary training method alongside Fine-Tuning and LoRA

This module provides:
- Prompt templates and optimization
- Few-shot learning without training
- Instruction tuning strategies
- System prompt caching (Anthropic style)
- Prompt chaining and composition
- Zero-shot and few-shot techniques
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


class PromptTemplate:
    """A reusable prompt template with variables."""
    
    def __init__(self, name: str, template: str, variables: List[str], category: str = "general"):
        self.name = name
        self.template = template
        self.variables = variables
        self.category = category
        self.usage_count = 0
        self.created_at = datetime.now().isoformat()
    
    def format(self, **kwargs) -> str:
        """Format the template with given variables."""
        self.usage_count += 1
        return self.template.format(**kwargs)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "template": self.template,
            "variables": self.variables,
            "category": self.category,
            "usage_count": self.usage_count,
            "created_at": self.created_at
        }


class InstructionSet:
    """A set of instructions for task-specific behavior."""
    
    def __init__(self, name: str, instructions: List[str], examples: List[Dict] = None):
        self.name = name
        self.instructions = instructions
        self.examples = examples or []
        self.created_at = datetime.now().isoformat()
    
    def to_prompt(self, include_examples: bool = True, max_examples: int = 5) -> str:
        """Convert to a full prompt."""
        prompt = "# Instructions\n\n"
        for i, instruction in enumerate(self.instructions, 1):
            prompt += f"{i}. {instruction}\n"
        
        if include_examples and self.examples:
            prompt += "\n# Examples\n\n"
            for i, example in enumerate(self.examples[:max_examples], 1):
                prompt += f"Example {i}:\n"
                if "input" in example:
                    prompt += f"Input: {example['input']}\n"
                if "output" in example:
                    prompt += f"Output: {example['output']}\n"
                prompt += "\n"
        
        return prompt


class PromptEngineeringManager:
    """
    Manages prompt engineering and instruction tuning.
    
    This is a training-free method that relies on:
    - Well-crafted prompts
    - Few-shot learning
    - System instructions
    - Cached examples
    """
    
    def __init__(self, storage_dir: str = "models/prompts"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self.templates: Dict[str, PromptTemplate] = {}
        self.instruction_sets: Dict[str, InstructionSet] = {}
        self.cached_prompts: Dict[str, str] = {}
        
        # Load built-in templates
        self._initialize_builtin_templates()
        self._initialize_architectural_instructions()
        
        logger.info("Prompt Engineering Manager initialized")
    
    def _initialize_builtin_templates(self):
        """Initialize built-in prompt templates."""
        
        # Architectural calculation template
        self.add_template(
            name="arch_calculation",
            template="""You are KURDO-AI, an expert architectural calculator.

Task: {task}
Given: {given_values}
Required: {required_output}

Show your calculation steps clearly.
Use appropriate units (metric: meters, square meters, cubic meters).
Provide practical recommendations when relevant.

Answer:""",
            variables=["task", "given_values", "required_output"],
            category="architecture"
        )
        
        # Code generation template
        self.add_template(
            name="code_generation",
            template="""You are KURDO-AI, an expert programmer.

Language: {language}
Task: {task}
Requirements:
{requirements}

Generate clean, well-commented code that follows best practices.

Code:""",
            variables=["language", "task", "requirements"],
            category="programming"
        )
        
        # Analysis template
        self.add_template(
            name="technical_analysis",
            template="""You are KURDO-AI, a technical analysis expert.

Subject: {subject}
Context: {context}
Analysis Type: {analysis_type}

Provide a detailed, structured analysis with:
1. Key findings
2. Technical details
3. Recommendations
4. Potential issues

Analysis:""",
            variables=["subject", "context", "analysis_type"],
            category="analysis"
        )
        
        # Design review template
        self.add_template(
            name="design_review",
            template="""You are KURDO-AI, an architectural design reviewer.

Project: {project_name}
Design Element: {design_element}
Standards: {applicable_standards}

Review for:
- Code compliance
- Structural integrity
- Practicality
- Cost-effectiveness
- Safety

Review:""",
            variables=["project_name", "design_element", "applicable_standards"],
            category="architecture"
        )
        
        # Translation template
        self.add_template(
            name="technical_translation",
            template="""You are KURDO-AI, a technical translator.

Source Language: {source_lang}
Target Language: {target_lang}
Domain: {domain}

Translate the following technical content accurately, preserving:
- Technical terms
- Measurements and units
- Structural meaning

Text to translate:
{text}

Translation:""",
            variables=["source_lang", "target_lang", "domain", "text"],
            category="translation"
        )
    
    def _initialize_architectural_instructions(self):
        """Initialize architectural instruction sets."""
        
        # Room calculation instructions
        room_calc_instructions = InstructionSet(
            name="room_calculations",
            instructions=[
                "Always specify units (meters, square meters, cubic meters)",
                "Show calculation steps: formula → substitution → result",
                "For area: A = length × width",
                "For volume: V = length × width × height",
                "Round to 2 decimal places for practical use",
                "Provide context (e.g., 'adequate for bedroom', 'requires ventilation')"
            ],
            examples=[
                {
                    "input": "محاسبه مساحت اتاق 5x4 متر",
                    "output": "مساحت = طول × عرض\nمساحت = 5 × 4 = 20 متر مربع\n\nاین مساحت برای یک اتاق خواب استاندارد مناسب است."
                },
                {
                    "input": "Calculate volume of room 6m × 4m with 2.8m height",
                    "output": "Volume = length × width × height\nVolume = 6 × 4 × 2.8 = 67.2 cubic meters\n\nThis volume requires adequate ventilation for residential use."
                }
            ]
        )
        self.add_instruction_set(room_calc_instructions)
        
        # Material estimation instructions
        material_instructions = InstructionSet(
            name="material_estimation",
            instructions=[
                "State standard ratios (e.g., 60 bricks per square meter)",
                "Calculate base requirement first",
                "Add 5-10% waste factor",
                "Provide practical ordering advice",
                "Consider standard package sizes"
            ],
            examples=[
                {
                    "input": "چند آجر برای دیوار 10 متری با ارتفاع 3 متر؟",
                    "output": "مساحت دیوار = 10 × 3 = 30 متر مربع\nآجر استاندارد = 60 عدد در هر متر مربع\nآجر مورد نیاز = 30 × 60 = 1,800 عدد\nبا ضریب اتلاف 10% = 1,980 عدد\n\nتوصیه: سفارش 2,000 آجر"
                }
            ]
        )
        self.add_instruction_set(material_instructions)
        
        # Code compliance instructions
        code_instructions = InstructionSet(
            name="building_codes",
            instructions=[
                "Reference specific code sections (e.g., مبحث 19)",
                "State minimum and recommended values",
                "Explain reasoning behind requirements",
                "Mention variations by region if applicable",
                "Always prioritize safety"
            ],
            examples=[
                {
                    "input": "حداقل ارتفاع سقف آپارتمان مسکونی؟",
                    "output": "طبق مبحث 19 مقررات ملی ساختمان:\n- اتاق‌های اصلی: حداقل 2.4 متر\n- راهرو: حداقل 2.1 متر\n- سرویس‌های بهداشتی: حداقل 2.1 متر\n\nتوصیه: ارتفاع 2.6-2.8 متر برای احساس فضای بهتر"
                }
            ]
        )
        self.add_instruction_set(code_instructions)
    
    def add_template(self, name: str, template: str, variables: List[str], category: str = "general") -> PromptTemplate:
        """Add a new prompt template."""
        pt = PromptTemplate(name, template, variables, category)
        self.templates[name] = pt
        self._save_template(pt)
        logger.info(f"Added template: {name}")
        return pt
    
    def add_instruction_set(self, instruction_set: InstructionSet):
        """Add a new instruction set."""
        self.instruction_sets[instruction_set.name] = instruction_set
        logger.info(f"Added instruction set: {instruction_set.name}")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self.templates.get(name)
    
    def list_templates(self, category: Optional[str] = None) -> List[str]:
        """List all templates, optionally filtered by category."""
        if category:
            return [name for name, tmpl in self.templates.items() if tmpl.category == category]
        return list(self.templates.keys())
    
    def create_few_shot_prompt(
        self,
        task_description: str,
        examples: List[Dict[str, str]],
        current_input: str,
        max_examples: int = 5,
        include_reasoning: bool = False
    ) -> str:
        """
        Create a few-shot learning prompt.
        
        Args:
            task_description: What the AI should do
            examples: List of {"input": "...", "output": "..."}
            current_input: The new input to process
            max_examples: Maximum number of examples to include
            include_reasoning: Include chain-of-thought reasoning
        
        Returns:
            Formatted few-shot prompt
        """
        prompt = f"# Task\n{task_description}\n\n"
        
        if include_reasoning:
            prompt += "Show your reasoning step-by-step.\n\n"
        
        prompt += "# Examples\n\n"
        
        for i, example in enumerate(examples[:max_examples], 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {example['input']}\n"
            
            if include_reasoning and "reasoning" in example:
                prompt += f"Reasoning: {example['reasoning']}\n"
            
            prompt += f"Output: {example['output']}\n\n"
        
        prompt += "# Your Turn\n\n"
        prompt += f"Input: {current_input}\n"
        
        if include_reasoning:
            prompt += "Reasoning: "
        else:
            prompt += "Output: "
        
        return prompt
    
    def create_chain_of_thought_prompt(
        self,
        problem: str,
        domain: str = "general"
    ) -> str:
        """
        Create a chain-of-thought prompt for complex reasoning.
        
        Args:
            problem: The problem to solve
            domain: Problem domain
        
        Returns:
            CoT prompt
        """
        prompt = f"""You are KURDO-AI, an expert in {domain}.

Problem: {problem}

Solve this step-by-step:
1. Understand: What is being asked?
2. Identify: What information do we have?
3. Plan: What approach should we use?
4. Calculate: Work through the solution
5. Verify: Does the answer make sense?
6. Conclude: State the final answer clearly

Let's work through this:

"""
        return prompt
    
    def create_role_based_prompt(
        self,
        role: str,
        expertise: List[str],
        task: str,
        constraints: Optional[List[str]] = None
    ) -> str:
        """
        Create a role-based prompt.
        
        Args:
            role: AI role (e.g., "structural engineer")
            expertise: List of expertise areas
            task: The task to perform
            constraints: Optional constraints
        
        Returns:
            Role-based prompt
        """
        prompt = f"You are KURDO-AI, a professional {role}.\n\n"
        prompt += "Your expertise includes:\n"
        for exp in expertise:
            prompt += f"- {exp}\n"
        prompt += "\n"
        
        if constraints:
            prompt += "Important constraints:\n"
            for constraint in constraints:
                prompt += f"- {constraint}\n"
            prompt += "\n"
        
        prompt += f"Task: {task}\n\n"
        prompt += "Provide a professional, detailed response:\n\n"
        
        return prompt
    
    def create_cached_system_prompt(
        self,
        system_role: str,
        training_examples: List[Dict],
        max_examples: int = 20
    ) -> Dict[str, Any]:
        """
        Create a cached system prompt (Anthropic style).
        
        This is cost-effective for repeated use with similar tasks.
        
        Args:
            system_role: System role description
            training_examples: Examples to cache
            max_examples: Maximum examples to include
        
        Returns:
            Cached prompt structure
        """
        cached_content = f"System Role: {system_role}\n\n"
        cached_content += "# Training Examples\n\n"
        cached_content += "Learn from these examples:\n\n"
        
        for i, example in enumerate(training_examples[:max_examples], 1):
            cached_content += f"Example {i}:\n"
            if "input" in example:
                cached_content += f"Q: {example['input']}\n"
            elif "prompt" in example:
                cached_content += f"Q: {example['prompt']}\n"
            
            if "output" in example:
                cached_content += f"A: {example['output']}\n"
            elif "completion" in example:
                cached_content += f"A: {example['completion']}\n"
            cached_content += "\n"
        
        cached_content += "Now respond to user queries following these patterns.\n"
        
        cache_id = f"cached_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.cached_prompts[cache_id] = cached_content
        
        return {
            "cache_id": cache_id,
            "cached_content": cached_content,
            "num_examples": min(len(training_examples), max_examples),
            "estimated_tokens": len(cached_content.split()),
            "usage": "Use this cached content as system message in API calls"
        }
    
    def optimize_prompt(
        self,
        original_prompt: str,
        optimization_strategy: str = "concise"
    ) -> Dict[str, Any]:
        """
        Optimize a prompt for better performance.
        
        Strategies:
        - concise: Remove redundancy
        - detailed: Add more context
        - structured: Improve formatting
        - directive: Make instructions clearer
        
        Args:
            original_prompt: The prompt to optimize
            optimization_strategy: Strategy to use
        
        Returns:
            Optimized prompt and metadata
        """
        optimizations = {
            "concise": self._optimize_concise,
            "detailed": self._optimize_detailed,
            "structured": self._optimize_structured,
            "directive": self._optimize_directive
        }
        
        if optimization_strategy not in optimizations:
            return {
                "status": "error",
                "message": f"Unknown strategy: {optimization_strategy}",
                "available_strategies": list(optimizations.keys())
            }
        
        optimized = optimizations[optimization_strategy](original_prompt)
        
        return {
            "original": original_prompt,
            "optimized": optimized,
            "strategy": optimization_strategy,
            "original_length": len(original_prompt),
            "optimized_length": len(optimized),
            "reduction": f"{(1 - len(optimized) / len(original_prompt)) * 100:.1f}%"
        }
    
    def _optimize_concise(self, prompt: str) -> str:
        """Make prompt more concise."""
        # Remove excessive whitespace
        lines = [line.strip() for line in prompt.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def _optimize_detailed(self, prompt: str) -> str:
        """Add more detail to prompt."""
        detailed = f"""Task Context:
{prompt}

Requirements:
- Be accurate and precise
- Show your work/reasoning
- Use appropriate units and formatting
- Provide practical recommendations

Response:"""
        return detailed
    
    def _optimize_structured(self, prompt: str) -> str:
        """Improve prompt structure."""
        structured = f"""# Task
{prompt}

# Instructions
1. Analyze the requirements carefully
2. Provide a structured response
3. Include relevant details and examples
4. Format output clearly

# Response
"""
        return structured
    
    def _optimize_directive(self, prompt: str) -> str:
        """Make instructions more directive."""
        directive = f"""INSTRUCTION: {prompt}

You must:
- Follow the instruction exactly
- Provide complete information
- Use clear, professional language
- Format output appropriately

BEGIN RESPONSE:
"""
        return directive
    
    def compare_with_training_methods(self) -> Dict[str, Any]:
        """
        Compare prompt engineering with other training methods.
        
        Returns:
            Detailed comparison
        """
        return {
            "prompt_engineering": {
                "type": "Training-Free",
                "setup_time": "Minutes",
                "cost": "$0 (inference only)",
                "gpu_required": False,
                "quality": "Good (depends on prompt quality)",
                "flexibility": "Very High",
                "best_for": [
                    "Quick prototypes",
                    "No training data",
                    "Rapid iteration",
                    "Zero-shot tasks",
                    "Cost-sensitive applications"
                ],
                "limitations": [
                    "Token limits (context window)",
                    "Repetitive examples increase cost",
                    "Less consistent than fine-tuning",
                    "Requires prompt engineering skills"
                ],
                "techniques": [
                    "Zero-shot prompting",
                    "Few-shot learning",
                    "Chain-of-thought",
                    "Role-based prompts",
                    "Instruction tuning",
                    "Prompt caching"
                ]
            },
            "fine_tuning": {
                "type": "Full Training",
                "setup_time": "Hours to Days",
                "cost": "$10-100+",
                "gpu_required": True,
                "quality": "Excellent",
                "flexibility": "Low (requires retraining)",
                "best_for": [
                    "Production deployments",
                    "Consistent behavior",
                    "Large-scale applications",
                    "Specialized domains"
                ]
            },
            "lora": {
                "type": "Parameter-Efficient Training",
                "setup_time": "30min-3hours",
                "cost": "$0 (local)",
                "gpu_required": True,
                "quality": "Very Good",
                "flexibility": "Medium (multiple adapters)",
                "best_for": [
                    "Multiple tasks",
                    "Limited GPU",
                    "Fast iteration",
                    "Cost-effective training"
                ]
            },
            "recommendation": {
                "use_prompt_engineering_when": [
                    "No training data available",
                    "Need immediate results",
                    "Budget is limited",
                    "Task changes frequently",
                    "Prototyping phase"
                ],
                "use_fine_tuning_when": [
                    "Have quality training data (500+ examples)",
                    "Need consistent behavior",
                    "Production deployment",
                    "Budget allows"
                ],
                "use_lora_when": [
                    "Have training data (50-500 examples)",
                    "Need multiple task-specific models",
                    "Limited GPU memory",
                    "Want fast training"
                ],
                "hybrid_approach": [
                    "Start with prompt engineering for prototyping",
                    "Collect real-world data",
                    "Train LoRA adapter for common tasks",
                    "Use prompt engineering for edge cases",
                    "Fall back to fine-tuning for production"
                ]
            }
        }
    
    def _save_template(self, template: PromptTemplate):
        """Save template to disk."""
        filepath = os.path.join(self.storage_dir, f"{template.name}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(template.to_dict(), f, indent=2, ensure_ascii=False)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_templates": len(self.templates),
            "templates_by_category": self._count_by_category(),
            "total_instruction_sets": len(self.instruction_sets),
            "cached_prompts": len(self.cached_prompts),
            "most_used_templates": self._get_most_used_templates(5)
        }
    
    def _count_by_category(self) -> Dict[str, int]:
        """Count templates by category."""
        counts = {}
        for template in self.templates.values():
            counts[template.category] = counts.get(template.category, 0) + 1
        return counts
    
    def _get_most_used_templates(self, limit: int = 5) -> List[Dict]:
        """Get most frequently used templates."""
        sorted_templates = sorted(
            self.templates.values(),
            key=lambda t: t.usage_count,
            reverse=True
        )
        return [
            {"name": t.name, "usage_count": t.usage_count, "category": t.category}
            for t in sorted_templates[:limit]
        ]

    def generate_prompt(
        self,
        query: str,
        task_type: Optional[str] = None,
        template_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a structured prompt for a query.

        Args:
            query: The user query
            task_type: Type of task (architectural, structural, etc.)
            template_name: Optional specific template to use
            **kwargs: Additional parameters for template

        Returns:
            Dictionary with prompt and metadata
        """
        # Select template
        if template_name and template_name in self.templates:
            template = self.templates[template_name]
        elif task_type:
            # Try to find template by task type
            category_map = {
                "architectural": "arch_calculation",
                "structural": "technical_analysis",
                "code": "code_generation",
                "design": "design_review"
            }
            template_name = category_map.get(task_type.lower(), "technical_analysis")
            template = self.templates.get(template_name)
        else:
            template = None

        # Generate prompt
        if template:
            try:
                # Fill template with query and kwargs
                template_vars = {k: kwargs.get(k, query) for k in template.variables}
                if "task" in template_vars and template_vars["task"] == query:
                    template_vars["task"] = query
                prompt = template.format(**template_vars)
            except Exception:
                # Fallback if template fails
                prompt = f"Task: {query}\n\nProvide a detailed response:"
        else:
            # Simple prompt without template
            prompt = f"Task: {query}\n\nProvide a detailed, structured response:"

        return {
            "status": "success",
            "prompt": prompt,
            "template_used": template.name if template else "default",
            "prompt_length": len(prompt),
            "query": query
        }


# Global instance
prompt_engineering_manager = PromptEngineeringManager()
