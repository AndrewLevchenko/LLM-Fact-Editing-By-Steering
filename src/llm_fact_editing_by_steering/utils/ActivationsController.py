import torch
from typing import Literal
from transformers import AutoModelForCausalLM, AutoTokenizer

class ActivationsController:
    def __init__(self, model:AutoModelForCausalLM, tokenizer:AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def get_activations(self, prompt: str, which_token: Literal['mean','last_token','all_tokens']='last_token',normalize=True) -> dict:
        """Принимает один(!) промпт, возвращает dict {номер слоя:тензор активаций}. Если which_token = last_token, возвращает эмбеддинг последнего токена, если mean - усредняет эмбеддинги по токенам, если all_tokens - выдает эмбеддинги по всем токенам).
        При last_token и mean возвращает тензор размерностью [hidden_dim], при all_tokens - [seq_len,hidden_dim] """

        activations= {}
        input = self.tokenizer(prompt,return_tensors="pt",).to(self.device)
        with torch.no_grad():
            outputs = self.model(**input,output_hidden_states=True)

        n_layers = len(outputs.hidden_states)

        for i in range(1, n_layers): # от 1 - потому что 0 - это эмбеддинги
            # среднее между токенами
            if which_token == 'mean':
                activations[i-1] = outputs.hidden_states[i].squeeze().mean(dim=0)#.detach().cpu()
            # последний токен
            elif which_token == 'last_token':
                activations[i-1] = outputs.hidden_states[i].squeeze()[-1]#.detach().cpu()
            # все токены
            elif which_token == 'all_tokens':
                activations[i-1] = outputs.hidden_states[i].squeeze()#.detach().cpu()
        if normalize:
            self._norm_activations(activations)
        return activations

    def activations_diff(self, prompt1: str, prompt2:str, which_token: Literal['mean','last_token','all_tokens']='last_token', normalize=True) -> dict:
        """Принимает два промпта, возвращает dict {номер слоя: разность тензоров активаций}. Если which_token = last_token, возвращает разность эмбеддингов последнего токена, если mean - разность средних эмбеддингов по токенам, если all_tokens - разность эмбеддингов для всех токенов).
        При last_token и mean возвращает тензор размерностью [hidden_dim], при all_tokens - [seq_len,hidden_dim] """
        activations1 = self.get_activations(prompt1, which_token)
        activations2 = self.get_activations(prompt2, which_token)
        diff = {}
        for i in activations1.keys():
            diff[i] = activations2[i] - activations1[i]
        # нормируем
        if normalize:
            self._norm_activations(diff)
        return diff

    @staticmethod
    def _norm_activations(activations):
        for layer_ind,activation in activations.items():
            # если размерность [hidden_dim]
            if len(activation.shape) == 1:
                activations[layer_ind] = activation/torch.norm(activation)
            # если размерность [seq_len,hidden_dim]
            if len(activation.shape) == 2:
                for token_ind in range(activation.shape[0]):
                    activations[token_ind,layer_ind] = activation/torch.norm(activation)

