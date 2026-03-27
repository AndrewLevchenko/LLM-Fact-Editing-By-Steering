import torch
from llm_fact_editing_by_steering.editscontrollers.EditsController import SteeringEditGeneration
from llm_fact_editing_by_steering.hookscontrollers.CosineMultLastTokensHooksController_instruct import \
    CosineMultLastTokensHooksController_instruct
from llm_fact_editing_by_steering.model import instruct_generate_text
from llm_fact_editing_by_steering.model import generate_text
from llm_fact_editing_by_steering.hookscontrollers.CosineMultLastTokensHooksControllerV2 import CosineMultLastTokensHooksControllerV2
from llm_fact_editing_by_steering.utils.load_dataset import load_dataset
from llm_fact_editing_by_steering.utils.load_model import load_model

@torch.no_grad()
def compute_sequence_logprob_autoregressive(
    model,
    tokenizer,
    prompt: str,
    target: str,
) -> float:
    """
    Computes log P(target | prompt) token by token, so hooks applied to the
    last token affect the same positions as during generation.
    """
    model_device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)

    if len(target_ids) == 0:
        raise ValueError("Target tokenization is empty.")

    total_logprob = 0.0
    current_ids = prompt_ids.copy()

    for token_id in target_ids:
        input_ids = torch.tensor([current_ids], device=model_device)

        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

        next_token_logits = logits[0, -1]
        next_token_logprobs = torch.log_softmax(next_token_logits, dim=-1)

        total_logprob += next_token_logprobs[token_id].item()

        current_ids.append(token_id)

    return total_logprob

@torch.no_grad()
def compute_sequence_logprob_stats(
    model,
    tokenizer,
    prompt: str,
    target: str,
) -> dict:
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    total_logprob = compute_sequence_logprob_autoregressive(
        model, tokenizer, prompt, target
    )

    return {
        "logprob_sum": total_logprob,
        "num_target_tokens": len(target_ids),
        "logprob_mean": total_logprob / len(target_ids),
    }

@torch.no_grad()
def compute_edit_success_full_sequence(
    model,
    tokenizer,
    prompt: str,
    old_target: str,
    new_target: str,
) -> dict:
    """
    Compares full-sequence log-probabilities:
        success = log P(new_target | prompt) > log P(old_target | prompt)
    """
    old_logprob = compute_sequence_logprob_stats(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target=old_target,
    )["logprob_sum"]
    new_logprob = compute_sequence_logprob_stats(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target=new_target,
    )["logprob_sum"]

    success = new_logprob > old_logprob

    return {
        "prompt": prompt,
        "old_target": old_target,
        "new_target": new_target,
        "old_logprob": old_logprob,
        "new_logprob": new_logprob,
        "logprob_diff": new_logprob - old_logprob,
        "success": success,
    }


def estimate_steering_alpha_search(model,tokenizer,dataset,basic_alpha=0.01,start_from = 0, end_at=1000,silent=False,blacklist=None, alpha_step = 0.001, max_alpha = 1):
    i = 1

    strings_for_llm_judge = ''

    counter_edited = 0
    counter_not_edited = 0
    list_of_unknown_result =[]
    print(len(dataset['train']))

    seg = SteeringEditGeneration(model, tokenizer, CosineMultLastTokensHooksControllerV2,layers=range(20,27))

    for example in dataset['train']:
        prompt = example['prompt']
        subject = example['subject']
        relation = example['relation']
        object = example['target_true']
        object_edited = example['target_false']
        torch.cuda.empty_cache()

        truth_message = [{"role": "system", "content":  ""},
                         {"role": "user",   "content": prompt.replace('{}',subject)}
        ]
        #prompt = tokenizer.apply_chat_template(truth_message,tokenize=False)
        print("-"*len(prompt))
        print(f'{prompt}')
        alpha = basic_alpha

        seg.drop_all_edits()

        print("No steering:")
        print(generate_text(model, tokenizer, prompt, max_new_tokens=50, do_sample=False))



        #print(f"{i}) {alpha=} {relation.replace('{}',subject)}")
        while True:
            seg.drop_all_edits()
            seg.set_edit(subject,relation,object,object_edited,alpha=alpha)

            result = compute_edit_success_full_sequence(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                old_target=object,
                new_target=object_edited,
            )


            if result['success']:
                counter_edited += 1
                print(f"Steering, alpha = {alpha}:")
                print(generate_text(model, tokenizer, prompt, max_new_tokens=50, do_sample=False))
                print(f"SUCCESS RATE: {counter_edited} of {(counter_edited + counter_not_edited)}: {counter_edited/(counter_edited + counter_not_edited)*100}%")
                print(result)
                i+=1
                break
            else:
                alpha += alpha_step
                if alpha > max_alpha:
                    counter_not_edited += 1
                    print(f"FAILURE:")
                    print(result)
                    i+=1
                    break



            #
            # if object_edited.lower() in prompt_answer_tuple[1].lower() and object.lower() not in prompt_answer_tuple[1].lower():
            #     counter_edited+=1
            #     break
            # elif object.lower() in prompt_answer_tuple[1].lower() and object_edited.lower() not in prompt_answer_tuple[1].lower():
            #     alpha += alpha_step
            #     print(f'alpha is set to {alpha}')
            #     if alpha > max_alpha:
            #         counted_not_edited+=1
            #         break
            #     else:
            #         continue
            # else:
            #     alpha += alpha_step
            #     print(f'alpha is set to {alpha}')
            #     if alpha > max_alpha:
            #         #counted_not_edited+=1
            #         list_of_unknown_result.append(i)
            #         break


    #     judge_string = f"{i}) {alpha=:.2} {relation.replace('{}',subject)} <<<{prompt_answer_tuple[1].replace('\n','. ')}>>> truth: {object}, target: {object_edited}"
    #     strings_for_llm_judge+= judge_string
    #     if silent==False:
    #         print(judge_string)
    #         print(f'Succesful edits: {counter_edited} of {i}. Unsuccessful edits: {counted_not_edited} of {i}.')
    #         print('-'*10)
    #     i += 1
    # print(f'You should make a revision for {list_of_unknown_result}')
    return strings_for_llm_judge


###################
if __name__ == "__main__":
    model, tokenizer = load_model("Qwen/Qwen3.5-9B")
    model.generation_config.max_length = None # чтобы не выдавал warning Both `max_new_tokens` (=100) and `max_length`(=4096) seem to have been set.
    model.eval()
    dataset = load_dataset()
    estimate_steering_alpha_search(model,tokenizer,dataset,basic_alpha=0.1,alpha_step=0.05,max_alpha=1.2)