# from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("/extrahome0/HF_models/Qwen/Qwen3-1.7B")

# Configurae the sampling parameters (for thinking mode)
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)

# Initialize the vLLM engine
llm = LLM(model="/extrahome0/HF_models/Qwen/Qwen3-1.7B")
llm = LLM(model="/home/ma-user/work/aicc/model/Qwen3-1.7B")
# # Prepare the input to the model
# prompt = "Give me a short introduction to large language models."
# messages = [
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=False,  # Set to False to strictly disable thinking
# )
# print(f'text:{text}')
problem = "In terms of $\\pi$, what is the area of the circle defined by the equation $2x^2+2y^2+10x-6y-18=0$?"
# problem = "Give me a short introduction to large language models."
# '<|im_start|>user\nGive me a short introduction to large language models.<|im_end|>\n<|im_start|>assistant\n'
text_existing = "<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer><|im_end|>\n<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}. " + problem + "<|im_end|>\n<|im_start|>assistant\n"
text_with_system = "<|im_start|>system\nYou are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST BE enclosed within <think> and </think> tags. Please reason step by step, and put your final answer within \\boxed{}.<|im_start|>user\n" + problem + "<|im_end|>\n<|im_start|>assistant\n"
text_without_system = "<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}. " + problem + "<|im_end|>\n<|im_start|>assistant\n"
text_without_boxed = "<|im_start|>user\n" + problem + "<|im_end|>\n<|im_start|>assistant\n"

# Generate outputs
outputs_existing = llm.generate([text_existing], sampling_params)
outputs_with_system = llm.generate([text_with_system], sampling_params)
outputs_without_system = llm.generate([text_without_system], sampling_params)
outputs_without_boxed = llm.generate([text_without_boxed], sampling_params)

# Print the outputs.
for output_existing, output_with_system, output_without_system, output_without_boxed in zip(outputs_existing, outputs_with_system, outputs_without_system, outputs_without_boxed):
    print("*"*100+"existing"+"*"*100)
    prompt_existing = output_existing.prompt
    generated_text_existing = output_existing.outputs[0].text
    print(f"Prompt: {prompt_existing!r}")
    print("-"*100)
    print(f"Generated text: {generated_text_existing!r}")
    print("*"*100+"with_system"+"*"*100)
    prompt_with_system = output_with_system.prompt
    generated_text_with_system = output_with_system.outputs[0].text
    print(f"Prompt: {prompt_with_system!r}")
    print("-"*100)
    print(f"Generated text: {generated_text_with_system!r}")
    print("*"*100+"without_system"+"*"*100)
    prompt_without_system = output_without_system.prompt
    generated_text_without_system = output_without_system.outputs[0].text
    print(f"Prompt: {prompt_without_system!r}")
    print("-"*100)
    print(f"Generated text: {generated_text_without_system!r}")
    print("*"*100+"without_boxed"+"*"*100)
    prompt_without_boxed = output_without_boxed.prompt
    generated_text_without_boxed = output_without_boxed.outputs[0].text
    print(f"Prompt: {prompt_without_boxed!r}")
    print("-"*100)
    print(f"Generated text: {generated_text_without_boxed!r}")