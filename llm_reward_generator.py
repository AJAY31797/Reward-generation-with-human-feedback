import transformers
import trl
import peft
import torch
from peft import PeftModel

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-3B-Instruct"

# Download tokenizer and model
# This was the original model. For now, I am trying to evaluate over the updated model. 
# tokenizer = AutoTokenizer.from_pretrained(model_name)
"""model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)"""

# Loading the fine-tuned model to collect the preferences again. 
# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# Use the base model now
model = base_model

# Load LoRA adapters
# model = PeftModel.from_pretrained(base_model, "Add path to updated parameters") # Now collect preferences using this model.

tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "You are a reward function designer."},
    {"role": "user", "content": "Generate a reward JSON for construction scheduling..."}
]

def build_prompt(project_context: str):
    system_content = """You are an expert in designing reward functions for reinforcement learning in construction scheduling. You generate reward specifications in JSON format."""

    user_content = """Design a reward function for a reinforcement learning-based construction scheduler. 

    The project context is: {project_context}

    OUTPUT FORMAT (JSON):
    {{
    "step_terms": [
        {{"channel": "time|cost", "source": "variable_name", "shape": "linear|quadratic|sqrt|log", "weight": number}}
    ],
    "terminal_terms": [
        {{"channel": "time|cost", "source": "variable_name", "shape": "linear|quadratic|sqrt|log", "weight": number}}
    ],
    "apply_terminal_only_if_complete": true|false
    }}

    AVAILABLE VARIABLES IN THE REWARD FUNCTION:
    - incremental_cost: Cost incurred in this time step
    - total_episode_cost: Total cost incurred till this time step
    - timestep: Current time step of the agent
    - number_of_started_tasks: Number of tasks started in this time step
    - deltat: Schedule time increment between this time step and previous time step
    - current_day: Current schedule time (measured in days)
    - number_of_finished_tasks: Number of tasks completed in this time step
    - n_elements: Total number of tasks

    "channel" for AVAILABLE VARIABLES IN THE REWARD FUNCTION:
    - incremental_cost: cost
    - total_episode_cost: cost
    - timestep: time
    - number_of_started_tasks: time
    - deltat: time
    - current_day: time
    - number_of_finished_tasks: time
    - n_elements: time

    REQUIREMENTS:
    - Minimize project cost and completion time
    - Use more than two reward components in step_terms and terminal_terms, whenever needed. However, each "source" should contain only one variable.
    - Do not add two variables in a single "source". For example, 'number_of_finished_tasks + number_of_started_tasks' is not a valid source.
    - Use negative weights to penalize undesirable metrics and positive weights to reward desirable metrics
    - "channel" must be EXACTLY "time" or "cost"
    - "shape" must be EXACTLY one of: "linear", "quadratic", "sqrt", "log"
    - Use negative weights (e.g., -1.0) to penalize, positive weights (e.g., 1.0) to reward
    - Do NOT create shape names like "negative_log" - use shape="log" with weight=-1.0 instead
    - Output ONLY valid JSON, no explanations, no extra text
    
    Generate the reward function JSON:
    """

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

project_context = """A residential building construction project, involving 20 activities. Need to create a schedule that minimizes time and cost."""

def generate_multiple_prompts(project_context: str, n):
    reward_jsons = []

    for i in range(0,n):
        prompt = build_prompt(project_context)
        # Apply chat template
        text = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True # Adds special tokens that tell model "now it's your turn to respond"
        )

        # Tokenize and generate
        inputs = tokenizer(text, return_tensors="pt").to(model.device) # Sends the text to the tokenizer and builds the model input
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=1, do_sample=True)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = response.split("assistant")[-1].strip()

        reward_jsons.append(assistant_response)
    
    return reward_jsons, prompt
