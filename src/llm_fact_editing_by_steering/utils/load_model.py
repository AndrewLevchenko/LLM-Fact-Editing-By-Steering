from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_fact_editing_by_steering.model import (ModelConfig,
                                                load_model_and_tokenizer,
                                                generate_text, instruct_generate_text)

def load_model(model_name:str) -> (AutoModelForCausalLM,AutoTokenizer):
    if model_name == 'meta-llama/Llama-2-7b-chat-hf':
        config = ModelConfig(model_name="meta-llama/Llama-2-7b-chat-hf", hf_token=None, device_map="cuda",
                         torch_dtype="bfloat16", load_in_4bit=True, use_fast_tokenizer=False,
                         padding_side="left",
                         local_files_only=False)
    elif model_name == "Qwen/Qwen3.5-9B":
        config = ModelConfig(model_name=model_name, hf_token=None, device_map="cuda",
                             torch_dtype="float16", load_in_4bit=True, use_fast_tokenizer=False,
                             padding_side="left",
                             local_files_only=False)
    elif model_name == "t-tech/T-lite-it-2.1":
        config = ModelConfig(model_name="t-tech/T-lite-it-2.1", hf_token=None, device_map="cuda",
                             torch_dtype="float16", load_in_4bit=True, use_fast_tokenizer=False,
                             padding_side="left",
                             local_files_only=False)
    else:
        raise ValueError("Модели, которые поддерживаются: 1) meta-llama/Llama-2-7b-chat-hf 2) t-tech/T-lite-it-2.1 3) Qwen/Qwen3.5-9B")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model, tokenizer = load_model_and_tokenizer(config)

    load_message = f"{model_name} model is loaded..."
    print(load_message)
    print("-"*len(load_message))

    return model, tokenizer

#load_model("meta-llama/Llama-2-7b-chat-hf")
#load_model("Qwen/Qwen3.5-9B")
#load_model("t-tech/T-lite-it-2.1")