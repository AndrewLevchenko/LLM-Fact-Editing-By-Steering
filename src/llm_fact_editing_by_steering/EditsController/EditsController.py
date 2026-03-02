from typing import Union
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_fact_editing_by_steering.utils import ActivationsController
from llm_fact_editing_by_steering.hookscontrollers import HooksController
from llm_fact_editing_by_steering.hookscontrollers import CosineMultLastTokensActDiffController

class EditsController:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,layer_range):
        self.model = model
        self.tokenizer = tokenizer
        self.edits = list()
        self.act_controller = ActivationsController(model,tokenizer)
        self.layer_range = layer_range

    def add_edits(self, edits: list) -> list:
        for edit in edits:
            self.add_edit(edit)

    def add_edit(self, edit: dict):
        object_to_edited_object_vectors = self.act_controller.activations_diff(self.object, self.object_edited,'last_token')
        relation_activation = self.act_controller.get_activations(self.relation.replace('{}', ''),which_token='last_token')
        subject_activation = self.act_controller.get_activations(self.subject, which_token='last_token')
        subject_and_relation_activation = self.act_controller.get_activations(self.relation.replace('{}', self.subject),which_token='last_token')
        hooks_controller = CosineMultLastTokensActDiffController(self.model,self.tokenizer, object_to_edited_object_vectors,
                                                       subject_activation,relation_activation,subject_and_relation_activation,
                                                       layers = self.layer_range,alpha=self.alpha)


class SteeringEditGeneration:
    def __init__(self, model:AutoModelForCausalLM, tokenizer:AutoTokenizer, subject:str, relation:str, object:str, object_edited:str, act_controller:ActivationsController,hooks_controller_class:HooksController,alpha):
        self.model = model
        self.tokenizer = tokenizer
        self.subject = subject
        self.relation = relation
        self.object = object
        self.object_edited = object_edited
        self.object_to_edited_object_vectors = act_controller.activations_diff(self.object,self.object_edited,'last_token') # надо брать именно last_token
        self.relation_activation = act_controller.get_activations(self.relation.replace('{}',''),which_token='last_token')
        self.subject_activation = act_controller.get_activations(self.subject,which_token='last_token')
        self.subject_and_relation_activation = act_controller.get_activations(self.relation.replace('{}',self.subject),which_token='last_token')
        self.layer_range = range(18,25)
        self.alpha = alpha
        self.hooks_controller = hooks_controller_class(self.model,self.tokenizer,self.object_to_edited_object_vectors,
                                                       self.subject_activation,self.relation_activation,self.subject_and_relation_activation,
                                                       layers = self.layer_range,alpha=self.alpha)

    def generate_with_edit(self,prompts:Union[str,list[str]],max_tokens:int=25, print_to_console:bool=False):

        prompt_answer_tuple = self.hooks_controller.steering_generation(prompts, max_tokens=max_tokens,print_to_console=print_to_console)
        return prompt_answer_tuple

def estimate_steering(dataset,alpha=0,start_from = 0, end_at=200,silent=False,blacklist=None):
    i = 1

    strings_for_llm_judge = ''

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
        seg = SteeringEditGeneration(model_chat,tokenizer_chat,subject,relation,object,object_edited,act_controller,CosineMultLastTokensHooksControllerV2,alpha=alpha)
        prompt_answer_tuple = seg.generate_with_edit(prompt,max_tokens=50)
        judge_string = f"{i}) {prompt_answer_tuple[0]} <<<{prompt_answer_tuple[1].replace('\n','. ')}>>> truth: {object}, target: {object_edited}\n"
        strings_for_llm_judge+= judge_string
        if silent==False:
            print(judge_string)
            print('-'*10)
        i += 1

    return strings_for_llm_judge