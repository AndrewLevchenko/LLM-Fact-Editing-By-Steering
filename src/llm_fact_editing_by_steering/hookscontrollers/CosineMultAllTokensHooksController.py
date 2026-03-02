import torch
from llm_fact_editing_by_steering.hookscontrollers.HooksController import HooksController
from transformers import AutoModelForCausalLM, AutoTokenizer

class CosineMultAllTokensHooksController(HooksController):

    def __init__(self, model:AutoModelForCausalLM, tokenizer:AutoTokenizer, steering_vectors, object_activations, layers, alpha=0.05):
        super().__init__(model, tokenizer, steering_vectors, layers, alpha=alpha)
        self.object_activations = object_activations

    def _steering_hook_fn_factory(self,layer_ind):

        vector = self.steering_vectors[layer_ind].to(self.model.device)

        def cosine_mult_last_all_tokens_hook(model, input, output):
            batch_size, seq_len, hidden_size = output.shape

            for element in range(batch_size):
                for token in range(seq_len):
                    cosine_sim = torch.cosine_similarity(self.object_activations[layer_ind].view(-1), output[element,token].view(-1), dim=-1)

                    if cosine_sim > 0:
                        output[element, token] = output[element, token].view(-1) + self.alpha * vector.view(-1) * torch.norm(output[element, token]) *cosine_sim
        return cosine_mult_last_all_tokens_hook