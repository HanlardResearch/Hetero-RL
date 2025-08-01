from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("/extrahome0/HF_models/Qwen/Qwen3-1.7B")

# Configurae the sampling parameters (for thinking mode)
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)

# Initialize the vLLM engine
llm = LLM(model="/extrahome0/HF_models/Qwen/Qwen3-1.7B")

# Prepare the input to the model
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    # enable_thinking=False,  # Set to False to strictly disable thinking
)
print(f'text:{text}')
# text_existing = "<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer><|im_end|>\n<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}. Let $p(x)$ be a polynomial of degree 4 such that $p(55) = p(83) = p(204) = p(232) = 8$ and $p(103) = 13.$  Find\n\\[p(1) - p(2) + p(3) - p(4) + \\dots + p(285) - p(286).\\]<|im_end|>\n<|im_start|>assistant"
# text_without_system = "<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}. In terms of $\\pi$, what is the area of the circle defined by the equation $2x^2+2y^2+10x-6y-18=0$?<|im_end|>\n<|im_start|>assistant"
# Generate outputs
outputs = llm.generate([text], sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print("*"*100)
    print(f"Generated text: {generated_text!r}")