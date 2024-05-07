from human_eval.data import write_jsonl, read_problems
from llama_cpp import Llama

llm = Llama(
      model_path="/home/mooizz/code-llm/cpu-inference/starcoder2-7b-f16.gguf",
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)
# output = llm(
#        "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
#       stop=["\nclass", "\ndef", "\n#", "\n@", "print", "if", "```", "<file_sep>"], # Stop generating just before the model would generate a new question
# ) # Generate a completion, can also call create_completion
# print(output)


def generate_one_completion(prompt):
    output = llm(prompt, stop=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>"])
    return output["choices"][0]["text"]


problems = read_problems()


num_samples_per_task = 200
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)