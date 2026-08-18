---
license: apache-2.0
---
[中文文档](README_CN.md)

# Dataset Description

This dataset is used for training and testing PiSSA models, and has been converted into [tatsu-lab/alpaca](https://hf-mirror.com/datasets/tatsu-lab/alpaca) format for ease of use. The project contains multiple folders, each corresponding to the enhancement of different model capabilities.
## Mathematical Reasoning corresponds to the folder metamath.
Test data comes from [openai/gsm8k](https://hf-mirror.com/datasets/openai/gsm8k) and [hendrycks/MATH](https://hf-mirror.com/datasets/hendrycks/competition_math).
Training data comes from [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA), which expands the GSM8K and MATH training sets to 395k samples, improving data quality and achieving better performance.
## Code Generation corresponds to the folder python.
Test data comes from [openai/humaneval](https://hf-mirror.com/datasets/openai/openai_humaneval) and [google/mbpp](https://hf-mirror.com/datasets/google-research-datasets/mbpp).
Training data comes from [m-a-p/CodeFeedback](https://hf-mirror.com/datasets/m-a-p/CodeFeedback-Filtered-Instruction), which contains 156,526 samples. Since both Humaneval and MBPP focus on testing Python capabilities, other programming languages were filtered out, leaving 104,848 Python samples for training.
## Multi-turn Dialogue and Instruction Following corresponds to the folder conversation.
Test data comes from [lmsys/mt-bench](https://huggingface.co/spaces/lmsys/mt-bench).
Training data comes from [WizardLM/evol_instruct_196k](https://hf-mirror.com/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k). Due to licensing constraints, only 143k samples were used for training.

# Testing Methodology

First, follow the [PiSSA](https://github.com/GraphPKU/PiSSA) README to set up the environment and download the dataset.
Training and testing commands for each task are provided in separate scripts, which can be executed directly to train and obtain results. Below, each script’s functions are explained in detail.

## Full Fine-tuning on MetaMath and Testing on GSM8K and MATH

Full fine-tuning is the most basic method. Use the following command to download the model and start training:
```
sh scripts/metamath_llama2_7b/run_full_finetune.sh
```
For downloading LLaMA models, replace `hf_***` with your HuggingFace token, and the model will be downloaded to the path specified by `--local-dir`:
```
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --token hf_*** --resume-download $BASE_MODEL --local-dir $BASE_MODEL
```
Considering training cost, Sections 5.1, 5.4, and 5.5 in the paper use only the first 100k samples for 1 epoch of training. To train on the full 395k samples, remove `:100000`. Sections 5.2 and 5.3 use the full dataset with 3 epochs of training. Multi-dataset training is also supported by listing the datasets to be used:
```
--sub_task metamath python conversation \
```
You can also adjust the number of samples for each dataset:
```
--sub_task metamath:100000 python:50000 conversation:50000 \
--num_train_epochs 1 \
```

The default training configuration uses `deepspeed zero 2`. For reduced GPU memory usage, use `zero 3`, though this increases communication overhead and may be slower. It is suitable for setups with limited hardware resources:
```
# --deepspeed configs/ds_config_zero2_no_offload.json \
--deepspeed configs/ds_config_zero3.json \
```
Since deepspeed conflicts with `CUDA_VISIBLE_DEVICES`, GPUs must be specified using `--include=localhost:0,1,2,3,4,5,6,7`. To align training results, set the GPU count, per-GPU batch size, and gradient accumulation steps to ensure a total batch size of 128. The calculation is as follows:
```
batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
```
Example:
```
--include=localhost:0,1,2,3,4,5,6,7
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
```
Testing

The test process uses the vllm framework to complete all tasks within minutes. For each task:
 - Use `utils/gen_vllm.py` to generate answers for test questions and save them as a JSON file.
 - Use `utils/test_acc.py` to compute accuracy based on the answers.

For the metamath folder, which contains both GSM8K and MATH test files, set --sub_task metamath:
```
python utils/gen_vllm.py --model $OUTPUT_PATH --sub_task metamath --output_file $OUTPUT_PATH/metamath_response.jsonl
python utils/test_acc.py --input_file $OUTPUT_PATH/metamath_response.jsonl
```
## Training with PiSSA on MetaMath and Testing on GSM8K and MATH

Run the following script to train and test with PiSSA:
```
sh scripts/metamath_llama2_7b/run_pissa.sh
```
PiSSA extracts the principal singular values and vectors from the base model, resulting in a modified base model called the Residual Model (containing non-principal components). Pre-initialized Residual Models and Adapters are available for download on [Hugging Face Collections](https://huggingface.co/collections/fxmeng):
```
RES_MODEL="fxmeng/PiSSA-Llama-2-7b-r128"
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download $RES_MODEL --local-dir $RES_MODEL
```
If a Residual Model exists locally, it will be used directly. Otherwise, PiSSA initialization is performed automatically:
```
if [ -e $RES_MODEL ]; then
    echo "Use pre-initialized residual model."
else
    echo "Perform PiSSA initialization by myself."
    python utils/init_pissa.py --base_model_path $BASE_MODEL --output_dir $RES_MODEL --init_weights pissa_niter_16 --lora_r 128 --lora_alpha 128 --lora_dropout 0 --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
fi
```
During initialization:
 - `--init_weights` True uses standard LoRA initialization.
 - `--init_weights` pissa uses PiSSA initialization.
 - `pissa_niter_16` enables fast singular value decomposition (SVD). For optimal speed and accuracy, set `niter=16`.

Residual Models and Adapters share the same structure as standard models and LoRA components, respectively. Specify the appropriate paths for seamless integration:
```
--model_name_or_path $RES_MODEL \
--adapter_name_or_path "pissa_init" \

model = PeftModel.from_pretrained(model, script_args.model_name_or_path, subfolder=script_args.adapter_name_or_path, is_trainable=True)
```
To simplify testing, merge the Adapter into the Residual Model after training:
```
--merge True \

python utils/gen_vllm.py --model $OUTPUT_PATH --sub_task metamath --output_file $OUTPUT_PATH/metamath_response.jsonl
python utils/test_acc.py --input_file $OUTPUT_PATH/metamath_response.jsonl
```
## Using QPiSSA to Quantize Base Models and Train on MetaMath, then Test on GSM8K and MATH

Run the following script to perform training and testing with QPiSSA:
```
sh scripts/metamath_llama2_7b/run_pissa.sh
```
QPiSSA supports directly quantizing the Residual Model into nf4 format by setting `--bits 4`. However, performing multiple passes of SVD decomposition significantly reduces quantization errors (reducing by approximately 20%, inspired by LoftQ). Pre-processed quantized models (tagged 4-bit-5iter) are available for download on Hugging Face Collections.

To customize, we recommend setting `--iter 5` for a five-pass SVD process:
```
python utils/init_qpissa.py --base_model_dir $BASE_MODEL --output_path $RES_MODEL --rank 128 --iter 5 --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
```
Testing: Combining Residual Model and Adapter

Since vllm does not support nf4 quantization, you need to convert the Residual Model from nf4 to bf16 for testing:
```
python utils/nf4_to_bf16.py --base_model_path $BASE_MODEL --quant_model_path $RES_MODEL/nf4 --output_path $RES_MODEL/bf16
```
Once converted, merge the Residual Model and Adapter. Unlike standard PiSSA, where merging can occur directly after training, QPiSSA-trained models need to first convert nf4 parameters to bf16 to prevent accuracy loss:
```
python utils/merge_adapter.py --base_model $RES_MODEL --adapter $OUTPUT_PATH/checkpoint-819/ --output_path $OUTPUT_PATH
```
Training on Code Tasks and Testing on Humaneval and MBPP

utils/gen_vllm.py also supports code-related tasks. Simply switch the task to --sub_task python. Use it to quickly generate answers for Humaneval and MBPP test sets. Post-process the outputs with utils/code_process.py, then compute the accuracy using evalplus:
```
python utils/gen_vllm.py --model $OUTPUT_PATH --sub_task python --output_file $OUTPUT_PATH/python_response.jsonl
python utils/code_process.py --path $OUTPUT_PATH/python_response.jsonl
evalplus.evaluate --dataset humaneval --samples $OUTPUT_PATH/humaneval.jsonl
evalplus.evaluate --dataset mbpp --samples $OUTPUT_PATH/mbpp.jsonl
```
Note: evalplus will output accuracy for both Humaneval/Humaneval+ and MBPP/MBPP+, but the reported results use the standard Humaneval and MBPP metrics.

## Training on Conversation Tasks and Testing on MTBench

To train for dialogue tasks, switch the task to `--sub_task` conversation. However, `utils/gen_vllm.py` does not currently support MTBench testing. Refer to the llm_judge project for detailed MTBench testing instructions.

MTBench involves two-turn dialogues, but since the training set lacks multi-turn data, the reported results focus only on the first-turn performance. This simplifies integration into the testing pipeline, and future updates will add support for MTBench testing.

# Important Notes

The provided testing scripts are designed for rapid comparisons of different training methods and are not comprehensive evaluations of model capabilities. For more professional and detailed evaluations, please refer to specialized benchmarking projects.


# Citation

If the dataset is helpful for your work, would you be willing to cite our paper?
```
@article{meng2024pissa,
  title={Pissa: Principal singular values and singular vectors adaptation of large language models},
  author={Meng, Fanxu and Wang, Zhaohui and Zhang, Muhan},
  journal={arXiv preprint arXiv:2404.02948},
  year={2024}
}
```