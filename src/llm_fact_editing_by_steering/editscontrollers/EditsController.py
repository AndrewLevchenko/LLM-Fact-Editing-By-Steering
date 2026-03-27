from typing import Union
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_fact_editing_by_steering.hookscontrollers.HooksController import HooksController
from llm_fact_editing_by_steering.utils.ActivationsController import ActivationsController

class SteeringEditGeneration:
    def __init__(self, model:AutoModelForCausalLM, tokenizer:AutoTokenizer, hooks_controller_class:HooksController,layers = range(18,25)):
        self.model = model
        self.tokenizer = tokenizer
        self.act_controller = ActivationsController(model,tokenizer)
        self.hooks_controller_class = hooks_controller_class
        self.layer_range = layers
        self.edits = []
        self.hooks_controller=None

    def set_edit(self,subject:str, relation:str, object:str, object_edited:str,alpha:float):
        self.subject = subject
        self.relation = relation
        self.object = object
        self.object_edited = object_edited
        self.alpha = alpha
        self.object_to_edited_object_vectors = self.act_controller.activations_diff(self.object, self.object_edited,'last_token')  # надо брать именно last_token
        self.relation_activation = self.act_controller.get_activations(self.relation.replace('{}', ''),which_token='last_token')
        self.subject_activation = self.act_controller.get_activations(self.subject, which_token='last_token')
        self.subject_and_relation_activation = self.act_controller.get_activations(self.relation.replace('{}', self.subject), which_token='last_token')
        self.hooks_controller = self.hooks_controller_class(self.model,self.tokenizer,
            self.object_to_edited_object_vectors,
            self.subject_activation,self.relation_activation,self.subject_and_relation_activation,
            layers = self.layer_range,alpha=self.alpha)
        self.hooks_controller.register_hooks()

    def drop_all_edits(self):
        if self.hooks_controller!=None:
            self.hooks_controller.kill_hooks()