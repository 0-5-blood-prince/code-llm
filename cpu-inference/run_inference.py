import torch
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM


import seaborn as sns
import time
import sys



def perplexity(test, tokenizer, model, device):
  encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
  max_length = tokenizer.model_max_length
  stride = 512
  seq_len = encodings.input_ids.size(1)

  nlls = []
  prev_end_loc = 0
  for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc]
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss
        torch.cuda.empty_cache()

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

  ppl = torch.exp(torch.stack(nlls).mean())
  return ppl













# import datasets
# import evaluate

# task_type = "text-classification"

# model_id = "juliensimon/distilbert-amazon-shoe-reviews"

# dataset_id = "juliensimon/amazon-shoe-reviews"
# label_column = "labels"
# label_mapping = {
#     "LABEL_0": 0,
#     "LABEL_1": 1,
#     "LABEL_2": 2,
#     "LABEL_3": 3,
#     "LABEL_4": 4,
# }

# data = datasets.load_dataset(dataset_id, split="test")
# print(data)
# metric = evaluate.load("accuracy")
# evaluator = evaluate.evaluator(task_type)


# def evaluate_pipeline(pipeline):
#     results = evaluator.compute(
#         model_or_pipeline=pipeline,
#         data=data,
#         metric=metric,
#         label_column=label_column,
#         label_mapping=label_mapping,
#     )
#     return results


# print("*** Original model")
# classifier = transformers.pipeline(task_type, model_id)
# results = evaluate_pipeline(classifier)
# print(results)

# print("*** ONNX")

# from optimum.onnxruntime import ORTModelForSequenceClassification
# from optimum.pipelines import pipeline

# model = ORTModelForSequenceClassification.from_pretrained(
#     model_id, from_transformers=True
# )
# tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
# model.save_pretrained("./model_onnx")
# tokenizer.save_pretrained("./model_onnx")
# classifier_onnx = pipeline(task_type, model=model, tokenizer=tokenizer)
# results = evaluate_pipeline(classifier_onnx)
# print(results)

# print("*** ONNX optimizer")

# from optimum.onnxruntime import ORTOptimizer
# from optimum.onnxruntime.configuration import OptimizationConfig

# optimizer = ORTOptimizer.from_pretrained(model)
# optimizer.optimize(
#     OptimizationConfig(
#         optimization_level=99, # 1, 2 or 99
#     ),
#     save_dir="./model_onnx",
# )
# model_optimized = ORTModelForSequenceClassification.from_pretrained(
#     "./model_onnx", file_name="model_optimized.onnx"
# )
# classifier_optimized = pipeline(task_type, model=model_optimized, tokenizer=tokenizer)
# results = evaluate_pipeline(classifier_optimized)
# print(results)

# print("*** ONNX quantizer")

# from optimum.onnxruntime import ORTQuantizer
# from optimum.onnxruntime.configuration import AutoQuantizationConfig

# quantizer = ORTQuantizer.from_pretrained(model)
# qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
# quantizer.quantize(save_dir="./model_onnx", quantization_config=qconfig)
# model_quantized = ORTModelForSequenceClassification.from_pretrained(
#     "./model_onnx", file_name="model_quantized.onnx"
# )
# classifier_quantized = pipeline(task_type, model=model_quantized, tokenizer=tokenizer)
# results = evaluate_pipeline(classifier_quantized)
# print(results)

if __name__ == "__main__":
    args = sys.argv[1:]
    exp = ""
    # Check if arguments were provided
    if not args:
        print("No arguments provided.")
    else:
        # Process the arguments
        exp = args[0]
    if exp == "vanilla":
        dataset = load_dataset("flytech/python-codes-25k")
        perplexity_dataset = dataset["train"].select(range(1))

        model_id = "bigcode/starcoder2-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        vanilla_model = AutoModelForCausalLM.from_pretrained(
                model_id
                # attn_implementation="flash_attention_2"
            )

        start = time.time()
        input_ids = tokenizer.encode( "\n\n".join(perplexity_dataset["text"]), add_special_tokens=False, return_tensors="pt")
        output = vanilla_model(input_ids,
        use_cache=True)
        print(len(output.past_key_values))
        for key_value in output.past_key_values:
            print(key_value[0].shape, key_value[1].shape)
        output_sequences = vanilla_model.generate(
        input_ids=input_ids,
        max_length=20+len(input_ids[0]),
        use_cache=True
        )
        print(f'Time Taken {time.time()-start}')
    elif exp == "mixed_precision":
        dataset = load_dataset("flytech/python-codes-25k")
        perplexity_dataset = dataset["train"].select(range(1))

        model_id = "bigcode/starcoder2-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        vanilla_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                # attn_implementation="flash_attention_2"
            )

        start = time.time()
        ppl = perplexity(perplexity_dataset,tokenizer,vanilla_model,"cuda")
        print(ppl, f'Time Taken {time.time()-start}')
    elif exp == "onnx":
        dataset = load_dataset("flytech/python-codes-25k")
        perplexity_dataset = dataset["train"].select(range(1))

        model_id = "bigcode/starcoder2-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        onnx_model = ORTModelForCausalLM.from_pretrained(
            model_id,export=True
        )
        onnx_model.save_pretrained("./model_onnx")
        tokenizer.save_pretrained("./model_onnx")

        start = time.time()
        ppl = perplexity(perplexity_dataset,tokenizer,onnx_model,"cpu")
        print(ppl, f'Time Taken {time.time()-start}')

