import os
import gc
import sys
import time
import torch
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler,
    ExLlamaV2StreamingGenerator,
)


model = None
config = None
tokenizer = None
cache = None
streaming = None
model_name = None
model_load_time = None
results = None
model_url = None
readme_table = None

def print_red(msg):
    return "\033[91m"+f"{msg}"+"\033[0m"

def print_green(msg):
    return "\033[92m"+f"{msg}"+"\033[0m"

def free_mem():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

def unload_model():
    global model, config, tokenizer, cache
    if model:
        model.unload()
    model, config, tokenizer, cache = None, None, None, None
    free_mem()


def load_model(model_dir, max_seq_len, batch_size, cache_8bit=False, split=None, low_mem=False, flash_option=True):
    load_start_time = time.time()
    global model, config, tokenizer, cache, model_load_time
    unload_model()
    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.low_mem = low_mem
    config.no_flash_attn = flash_option
    config.prepare()

    model = ExLlamaV2(config)

    tokenizer = ExLlamaV2Tokenizer(config)

    try:
        model.load(split)
    except Exception as e:
        raise e
    if cache is None:
        try:
            if cache_8bit:
                cache = ExLlamaV2Cache_8bit(model, max_seq_len=max_seq_len, lazy=False, batch_size=batch_size)
            else:
                cache = ExLlamaV2Cache(model, max_seq_len=max_seq_len, lazy=False, batch_size=batch_size)
        except Exception as e:
            raise e
    load_end_time = time.time()
    model_load_time = load_end_time - load_start_time
    return True

def sampler_settings():
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.85
    settings.top_k = 50
    settings.top_p = 0.8
    settings.token_repetition_penalty = 1.15
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])
    return settings

def test_generator(prompt, max_new_tokens):
    print()
    print("--------------------------------")
    print("Generating: unbatched,")
    settings = sampler_settings()

    try:
        generator_class = ExLlamaV2StreamingGenerator if streaming else ExLlamaV2BaseGenerator
        generator = generator_class(model, cache, tokenizer)
        time_begin = time.time()
        output = generator.generate_simple(prompt, settings, max_new_tokens, seed=1234, token_healing=token_healing)
    except Exception as e:
        raise e
    time_end = time.time()
    time_total = time_end - time_begin
    return (output, time_total)

def test_gen_batch(max_new_tokens, prompt_batch, batch_size):
    print()
    print("--------------------------------")
    print("Generating: batched")
    cache.current_seq_len = 0
    settings = sampler_settings()
    print(f"Prompt processing, {model.config.max_seq_len - 1} tokens...")
    try:
        generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
        generator.warmup()
    except Exception as e:
        raise e
    try:
        print(f"Generating batch of {len(prompt_batch)} prompts")
        time_begin = time.time()
        output = generator.generate_simple(prompt_batch, settings, max_new_tokens, seed = 1234, token_healing = token_healing)
        time_total = time.time() - time_begin
    except Exception as e:
        print(print_red(f"Failed to generate: {e}"))

    return (output, time_total)


def generate_configurations(options):
    for low_mem in options:
        for streaming in options:
            yield low_mem, streaming

def execute_test(model_dir, max_seq_len, split, batch_size, low_mem, flash_option, cache_type, streaming, prompts, is_batch, max_new_tokens):
    start_time = time.time()
    unload_model()
    print_test_info(model_name, batch_size, low_mem, cache_type, streaming, is_batch, token_healing)
    try:
        load_model(model_dir, max_seq_len, cache_8bit=(cache_type == "Cache_8bit"), split=split, low_mem=low_mem, flash_option=flash_option, batch_size=batch_size)
    except Exception as e:
        raise e
    try:
        if is_batch:
            prompt_batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
            prompt_batch = prompt_batches[0]
            # for prompt_batch in prompt_batches:
            if len(prompt_batch) == batch_size:
                output, total_time = test_gen_batch(max_new_tokens, prompt_batch, batch_size=batch_size)
                throughput = batch_size * max_new_tokens / total_time
                print(print_green(f"Response generated in {total_time:.2f} seconds, {max_new_tokens} tokens, throughput using batch of {batch_size}: {batch_size * max_new_tokens / total_time:.2f} tokens/second"))
                random_index = random.randint(0, len(output) - 1)
                max_memory_allocated = torch.cuda.max_memory_allocated(device_index)
                log_experiment_results_to_csv_file(model_name, model_url, output[random_index], prompt_batch[random_index], batch_size, low_mem, cache_type, streaming, is_batch, token_healing, max_new_tokens, total_time, throughput, max_memory_allocated)
            else:
                print(print_red(f"Skipping batch of size {len(prompt_batch)}"))
        else:
            output, total_time = test_generator(prompts[0], max_new_tokens)
            throughput = max_new_tokens / total_time
            print(print_green(f"Response generated in {total_time:.2f} seconds, {max_new_tokens} tokens, throughput: {max_new_tokens / total_time:.2f} tokens/second"))
            max_memory_allocated = torch.cuda.max_memory_allocated(device_index)
            log_experiment_results_to_csv_file(model_name, model_url, output, prompts[0], batch_size, low_mem, cache_type, streaming, is_batch, token_healing, max_new_tokens, total_time, throughput, max_memory_allocated)
        end_time = time.time()
    except Exception as e:
        print(print_red(e))
    finally:
        unload_model()


def print_test_info(model_name, batch_size, low_mem, cache_type, streaming, is_batch, token_healing):
    test_type = "Batch" if is_batch else "Single"
    if cache_type == "Cache_8bit":
        cache_name = "8-bit"
    else:
        cache_name = "16-bit"
    info = (
        f"Test type:   {print_green(test_type)}\n"
        f"Model:       {model_name}\n"
        f"Batch Size:  {batch_size}\n"
        f"Low Mem:     {low_mem}\n"
        f"Cache Type:  {cache_name}\n"
        f"Streaming:   {streaming}\n"
        f"Token Heal:  {token_healing}\n"
    )
    print()
    print(info)

def log_experiment_results_to_csv_file(model_name, model_url, output_sample, input_prompt, batch_size, low_mem, cache_type, streaming, is_batch, token_healing, max_new_tokens, time_total, throughput, max_memory_allocated):
    global results
    test_type = "Batch" if is_batch else "Single"
    if cache_type == "Cache_8bit":
        cache_name = "8-bit"
    else:
        cache_name = "16-bit"
    if results is None:
        with open("results.csv", "w") as results:
            results.write("Model_Name, model_url, Batch_Size, Low_Mem, Cache_Type, Streaming, Is_Batch, Token_Healing, Max_New_Tokens, Time_Total, Tokens_per_second, VRAM_GB_used, Input_Prompt_Truncated, Output_Sample_Truncated\n")
    # Todo: input_prompt and output_sample are not csv safe. Need to escape commas and newlines
    input_prompt = input_prompt.replace(",", " ")
    input_prompt = input_prompt.replace("\n", " ")
    output_sample = output_sample.replace(",", " ")
    output_sample = output_sample.replace("\n", " ")
    with open("results.csv", "a") as results:
        results.write(f"{model_name}, {model_url}, {batch_size}, {low_mem}, {cache_name}, {streaming}, {is_batch}, {token_healing}, {max_new_tokens}, {time_total:.2f}, {throughput:.0f}, {max_memory_allocated / 1024 ** 3:.2f}, {input_prompt:100}, {output_sample:100}\n")
                    

prompts = ["Once you eliminate all the",
           "C++ is",
           "A bird in the hand is worth two in the bush, but",
           "Too many cooks spoil the",
           "A lynx is a type of",
           "Standing before the gates of",
            "from typing import List\n\n\ndef string_xor(a: str, b: str) -> str:\n    \"\"\" Input are two strings a and b consisting only of 1s and 0s.\n    Perform binary XOR on these inputs and return result also as a string.\n    >>> string_xor('010', '110')\n    '100'\n    \"\"\"\n",
            "\n\ndef truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n",
            "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
            "from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
            "Here's how to create a powerful love potion",
            "For once,",
            "The events of the American Civil War",
            "A bird in the hand is worth"
           ]

home_dir = os.path.expanduser("~")
models_directory = os.path.join(home_dir, "Models")

options = [True]
max_new_tokens = 400
batch_sizes=[1, 2, 4, 8]
max_seq_len = 1024
for model_dir in os.listdir(models_directory):
    model_name = model_dir
    model_full_path = os.path.join(models_directory, model_dir)
    model_url = os.popen("git -C " + model_full_path + " config --get remote.origin.url").read().strip()
    print(f"Model_url: {model_url}")
    if os.path.isdir(os.path.join(model_full_path, ".git")):
        print("Git repo found")
        branch_name = os.popen("git -C " + model_full_path + " branch --show-current").read().strip()
        if branch_name != "main":
            quantization = branch_name
            model_name = model_dir + "-" + branch_name
            print(f"model_name: {model_name}")
    
    for device_index in [0]: #normally use range(torch.cuda.device_count()): but set manually for testing
        print("\033[92m" + f"Running experiments on CUDA device {device_index}")
        torch.cuda.set_device(device_index)
        cuda_name = torch.cuda.get_device_name(device_index)
        print(torch.cuda.get_device_name(device_index))
        print(torch.cuda.get_device_properties(device_index))
        if device_index == 0: 
            split=[7.5,0.0]
        else:
            split=[0.0,7.5]
        for token_healing in [True]:
            for batch_size in batch_sizes:
                for test_config in generate_configurations(options):
                    low_mem, streaming = test_config
                    cache_type = "Cache"
                    if batch_size > 1:
                        execute_test(model_full_path, max_seq_len=max_seq_len, split=split, batch_size=batch_size, low_mem=low_mem, cache_type=cache_type, prompts=prompts, is_batch=True, max_new_tokens=max_new_tokens, flash_option=True, streaming=False)
                    else:
                        execute_test(model_full_path, max_seq_len=max_seq_len, split=split, batch_size=batch_size, low_mem=low_mem, cache_type=cache_type, prompts=prompts, is_batch=False, max_new_tokens=max_new_tokens, flash_option=True, streaming=False)