import torch
from llm_fact_editing_by_steering.hookscontrollers.HooksController import HooksController
from transformers import AutoModelForCausalLM, AutoTokenizer

class CosineMultLastTokensActDiffController(HooksController):

    def __init__(self, model:AutoModelForCausalLM, tokenizer:AutoTokenizer, object_to_edited_object_vectors, subject_activations, relation_activations,subject_and_relation_activations,layers, alpha):
        super().__init__(model, tokenizer, object_to_edited_object_vectors, layers, alpha=alpha)
        self.subject_activations = subject_activations
        self.relation_activations = relation_activations
        self.subject_and_relation_activations = subject_and_relation_activations
    def _steering_hook_fn_factory(self,layer_ind):

        vector = self.steering_vectors[layer_ind].to(self.model.device)

        alpha_decrease = (max(self.layers)-layer_ind)/(max(self.layers)-min(self.layers))

        def cosine_mult_last_token_hook(model, input, output):
            batch_size, seq_len, hidden_size = output.shape
            token=-1
            for element in range(batch_size):
                subject_cosine_sim = torch.cosine_similarity(self.subject_activations[layer_ind].view(-1), output[element,token].view(-1), dim=-1)
                relation_sim = torch.cosine_similarity(self.relation_activations[layer_ind].view(-1), output[element,token].view(-1), dim=-1)
                subject_and_relation_sim = torch.cosine_similarity(self.subject_and_relation_activations[layer_ind].view(-1), output[element,token].view(-1), dim=-1)
                if subject_and_relation_sim > 0:
                    output[element, token] = output[element, token].view(-1) + self.alpha * alpha_decrease * vector.view(-1) * torch.norm(output[element, token]) * subject_and_relation_sim

        return cosine_mult_last_token_hook