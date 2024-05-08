# Optimizing Code LLM Inference in GPUs and Commodity Hardware


This README contains a brief description of the whole project. The READMEs in subfolders cpu-inference and gpp-inference contain the details of experimentation done for each hardware.


## Problem Statement 

Large Language Models for Code have rapidly emerged as powerful assistants for writing and editing code. In this project, we are primarily interested in optimizing the inference of Code Generation in Large Language Models (LLMs), our goal is to enable efficient inference, allowing programmers to leverage these models on a single GPU or even on CPU. In the realm of GPU, We used Flash Attention 2 which helps in efficient attention computation as this step is the time-intensive component of the inference. In the realm of CPU hardware, we used llama.cpp and GGML format models to perform efficient inference on CPU.

<be>

Models Used: Starcoder2 7B

Evaluation Benchmark: HumanEval



## GPU Inference
In the first half of the experiments, we benchmark inference speeds on Starcoder 2 with and without Flash Attention. Then we benchmark the quantized version of Starcoder2 (quantization is achieved using advanced techniques like BitsandBytes and GPTQ). Later, we evaluate the performance of these quantized models using perplexity and pass@1 on Hu as metrics. More information about the methodology and results is present in gpu-inference README file.

![image](https://github.com/0-5-blood-prince/code-llm/assets/42780672/24c6730d-5504-4c84-af9f-1c14cfa154b0)


## CPU Inference
In these experiments, we first performed inference using torch-cpu environment and JIT optimization. Later, we used GGML model format to perform inference using llama.cpp. This resulted in a huge performance improvement. We further improve the inference speed by using quantization. See cpu-inference README for further details

GGML models: https://huggingface.co/Mooizz/starcoder2-gguf

![image](https://github.com/0-5-blood-prince/code-llm/assets/42780672/b18a652f-66d6-4ded-a7a9-d627b9c24f4a)



