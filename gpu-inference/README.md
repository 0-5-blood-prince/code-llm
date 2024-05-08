#GPU INFERENCE

We have conducted two experiments on huggingface implementation of Starcoder2 which supports different modes including enabling FlashAttention2, Quantization using different wrappers. The jupyter notebook Starcoder2_LLM contains all the code used to benckmark the Starcoder2 and a detailed explanation of every benchmark experiment is mentioned below:



## FLASH ATTENTION 2
To measure the speedup that we could achieve with flash attention, we measure the time taken for StarCoder2 (without and with Attention) in multiple settings. We use the time taken for one minibatch to measure the speedup. This is calculated in the warmup and benchmark function where we run 2 minibatches of warmup and then measure the time taken to infer 4 minibatches. This process is repeated 4 times to find the time taken for minibatch which is used to measure the speedup.

\\

### Starcoder2 using fp32 (without and with Flash Attention 2)
We have measured the time taken as mentioned above for Starcoder2 using FP32 without flash attention only (because the support for FP32 with flash attention is not available in huggingface). We have used the following combination of batch sizes and context lengths and their corresponding time measurements

||Time Taken (seconds)|
|:------------:|:----------------:|
| Batch Size \ Context Length | **4096**            |
| **2**         | 29.02          |
| **4**         | 58.02         |
| **8**         | OOM         |

\\

### Starcoder2 using bf16 (without and with Flash Attention 2)
We have measured the time taken as mentioned above for Starcoder2 using FP16 without and with flash attention. We have used the following combination of batch sizes and context lengths and their corresponding time measurements

\\

||Time Taken (seconds)| Without Flash Attention|||
|:------------:|:--------------:|:----------:|:------------:|:----------:|
| Batch Size \ Context Length | **4096**  | **8192** | **16384** | **18000** |
| **2**         |    11.05       |     22.75      |     49.04 | 55.20 |
| **4**         | 25.71         | 66.96 | OOM | OOM |
| **8**         | 53.32          | OOM | OOM | OOM |

\\

||Time Taken (seconds)| With Flash Attention|||
|:------------:|:--------------:|:----------:|:------------:|:----------:|
| Batch Size \ Context Length | **4096**  | **8192** | **16384** | **18000** |
| **2**         |    9.87       |   19.95    |  **43.80** | 45.26 |
| **4**         | 19.64        | **41.28** | OOM | OOM |
| **8**         | **40.06**          | OOM | OOM | OOM |

\\

||Speed up||||
|:------------:|:--------------:|:----------:|:------------:|:----------:|
| Batch Size \ Context Length | **4096**  | **8192** | **16384** | **18000** |
| **2**         |  1.12      |   1.14    |  1.19 | 1.21 |
| **4**         | 1.31       | **1.62** | OOM | OOM |
| **8**         | 1.31       | OOM | OOM | OOM |

\\

**Results:**
Now, we can calculate and compare the speedups with and without flash attention and below are some of our observations.

* From the speedup table, we see that as both batch size and context length increase, the speedup values increase.
* There's one important observation that we need to look for in the time taken table for flash attention, the values are almost the same when we look along diagonally from right to left, this is basically the time taken for inference for a fixed amount of memory (multiplication of batch size and context length) which proves the point made in the flash attention paper that the time taken increases linearly with the amount of data handled.
* When we compare the speed up obtained when used fp32 vs bf16 (without flash attention), we see that it is more than 2 times throughout.

\\

## Quantization

We have benckmarked the bf16, fp8 and nf4 implementations of the Starcoder2 (with flash attention enabled for all the quantizations), similar to the method used in flash attention 2, which is to measure the time taken for one minibatch. We have also measured the perplexity score on first 512 entries in flytech/python-codes-25k dataset in huggingface along with evaluating the pass@1 score on HumanEval dataset.

\\

### Brief introduction about the quantization methods used
The authors of Starcoder2 model have implemented 2 versions code CodeLLM in Huggingface (fp32 and bf16), where they have only quantized the weights of the Starcoder2 but not the activations. In order to use the StarCoder2 in fp8 and fp4 settings, we need to use the pre-defined wrappers in huggingface, we have used 2 wrappers to evaluate the CodeLLM: **BitsAndBytes** and **GPTQ**. BitsAndBytes achvieves quantization of LLMs in 8-bit using vector-wise quantization to quantize most features to 8-bits and separately treating outliers with 16-bit matrix multiplication (And it is PTQ) whereas it achieves 4-bit using a novel technique called QLORA (Quantized-Low Rank Adaptation) and it quantizes the model to 4-bits and inserts a small set of trainable low-rank adaptation (LoRA) weights to allow training.

\\

### Quantization using Bits And Bytes

We have measured the time taken for one minibatch, perplexity score and pass@1 values for bf16, fp8 and nf4 implementations. We have also measured the time taken for nf4 quantized Starcoder2 model with compute size of torch.float32 and torch.float16.

\\

||Time Taken (seconds)||||
|:------------:|:--------------:|:----------:|:------------:|:------------:|
| Batch Size \ Quantization | **bf16**  | **fp8** | **nf4 with torch.float16** | **nf4 with torch.float32**|
| **2**         |    9.87       |   13.28    |  13.31 | 101.56 |
| **4**         | 19.64        | 27.88 | 28.24 | 202.8 |
| **8**         | 40.06          | 57.96 | 55.64 | 401.16 |

\\

||Evaluation|||
|:------------:|:----------------:|:----------:|:------------:|
| Metric \ Quantization | **bf16**  | **fp8** | **nf4 with torch.float16** |
| **Perplexity**         | 1.107   |  1.704   |  1.714   |
| **pass@1**         | 0.359     |  0.335     |  0.329     |
