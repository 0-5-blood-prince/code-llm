from human_eval.data import write_jsonl, read_problems
from llama_cpp import Llama
import time



### Load the llama_cpp module
llm = Llama(
      model_path="/home/mooizz/starcoder2-gguf/starcoder2-7b-f16.gguf",
      n_ctx = 512, n_batch = 2048, n_keep = 0, n_threads = 8, echo=False)


### code completion task
def generate_one_completion(prompt):
    output = llm(prompt,
                # stop=["\n\n\n"]
                 stop=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>"],
                #  repeat_last_n = 64, 
                max_tokens=300,
                echo=False,
                 repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000,

        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temperature = 0.800)
    return output["choices"][0]["text"]
      
problems = read_problems()

num_samples_per_task = 1

start = time.time()
for _ in range(num_samples_per_task):
    for task_id in problems:
        completion = generate_one_completion(problems[task_id]["prompt"])
        print(completion)
        dict(task_id=task_id, completion=completion)


print(f'Total Time Taken {time.time()-start}')
write_jsonl("samples.jsonl", samples)
