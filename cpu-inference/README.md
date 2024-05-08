# CPU Inference of StarCoder-2 7B model

We used multiple optimizations to create an efficient Code LLM capable of running inference on CPU
They are the following
* Vanilla - running torch-cpu
* pytorch-cpu jit traced model
* Using a GGML model format for efficient inference, further we use quantization to reduce the memory footprint and inference speed.


Model: Starcoder2 7B Code generation model
Frameoworks and libraries used: pytorch, hugging face and llama.cpp


Optmization metric: Inference Speed
Evaluation metric: HumanEval Benchmark

Files
* trainer_code_gen.py, This file contains code to measure inference speed of a given prompt, with two modes Vanilla and JIT compiled model.
* starcoder2_download.py, This file contains the code to download huggingface pre-trained model weights in a particular directory


## pytorch-cpu only

Using torch-cpu instead of torch-cuda

``````
python trainer_code_gen.py --prompt {PROMPT} --cpu_only --max_length {MAX_TOKENS} 
``````


This command displays the decoded outputs of model when prompt with the given PROMPT

## pytorch-cpu JIT compiled model

A jit trace of a sample input to the model is used to create traced model. And the inference is done the JIT compiled model. 

`````` 
python trainer_code_gen.py --prompt {PROMPT} --cpu_only -jit 
``````


## GGML format

### llama.cpp

llama.cpp is 

### Converting Starcoder2 to GGML format

ref: https://github.com/ggerganov/llama.cpp/discussions/2948

Download Starcoder2 weights
``````
python starcoder2_download.py {FOLDER}
``````

Convert the model to f16 GGUF format
``````
python llama.cpp/convert.py {FOLDER} \
  --outfile starcoder-7b-q16.gguf \
``````


To perform Inference on the GGUF model

``````
cd llama.cpp && ./main -m ../starcoder2-gguf/starcoder2-7b-f16.gguf
 -n NUM_TOKENS 
--prompt "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n   ......."
``````
### Quantizing the GGUF Models


To get a 4-bit quantized model with Q4_0 type quantization
``````
./examples/quantize ../starcoder2-gguf/starcoder2-7b-f16.gguf ../starcoder2-gguf/starcoder2-7b-q4_0.gguf q4_0
``````

To get a 4-bit quantized model with Q4_K_M type quantization
``````
./examples/quantize ../starcoder2-7b-f16.gguf ../starcoder2-7b-q4_k_m.gguf q4_k_m
``````

To get a 4-bit quantized model with Q4_K_S type quantization
``````
./examples/quantize ../starcoder2-7b-f16.gguf ../starcoder2-7b-q4_k_s.gguf q4_k_s
``````


Quantized models: HF link: https://huggingface.co/Mooizz/starcoder2-gguf/tree

Evaluation on HumanEval Benchmark

Run the below command

``````
python benchmark-llama-gguf.py
``````


#### Inference Results with an example Prompt: 


<code>
from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n
</code>

with max_tokens=200


* Vanilla torch-CPU: 257.16 sec
* torch-jit (trace): 240.62 sec
* GGML(f16) with llama.cpp: 54 sec
* GGML(q4_k_m) with llama.cpp: 48.9 sec
* GGML(q4_k_s) with llama.cpp: 26.22 sec









