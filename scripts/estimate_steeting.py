import torch
from llm_fact_editing_by_steering.editscontrollers import EditsController
from llm_fact_editing_by_steering.utils import ActivationsController
from llm_fact_editing_by_steering.hookscontrollers import CosineMultLastTokensHooksControllerV2


def estimate_steering(model, tokenizer,dataset,alpha=0,start_from = 0, end_at=200,silent=False,blacklist=None):
    i = 1

    strings_for_llm_judge = ''
    act_controller = ActivationsController(model,tokenizer)
    for example in dataset['train']:
        if i < start_from or (blacklist is not None and i in blacklist):
            i+=1
            continue
        elif i > end_at:
            break
        prompt = example['prompt']
        subject = example['subject']
        relation = example['relation']
        object = example['target_true']
        object_edited = example['target_false']
        torch.cuda.empty_cache()
        seg = SteeringEditGeneration(model,tokenizer,subject,relation,object,object_edited,act_controller,CosineMultLastTokensHooksControllerV2,alpha=alpha)
        prompt_answer_tuple = seg.generate_with_edit(prompt,max_tokens=50)
        judge_string = f"{i}) {prompt_answer_tuple[0]} <<<{prompt_answer_tuple[1].replace('\n','. ')}>>> truth: {object}, target: {object_edited}\n"
        strings_for_llm_judge+= judge_string
        if silent==False:
            print(judge_string)
            print('-'*10)
        i += 1

    return strings_for_llm_judge
