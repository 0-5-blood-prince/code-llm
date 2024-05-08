# Democratize Code Generation from LLMs

We are primarily interested in optimizing the inference of Code Generation in Large Language Models (LLMs), our goal is to enable efficient inference which can be very useful in the software development context, allowing programmers to leverage these models on a single GPU or even on CPU. We have used Flash Attention 2 which helps in efficient attention computation as this step is the time-intensive component of the inference. In the first half of the experiments which is GPU inference, we benchmark inference on Starcoder 2 with and without Flash Attention. We also benchmark the quantized version of Starcoder2 (quantization is achieved using advanced techniques like BitsandBytes and GPTQ) and evaluate the performance of these quantized models using perplexity and pass@1 as metrics.


