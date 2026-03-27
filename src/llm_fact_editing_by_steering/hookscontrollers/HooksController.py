import torch
from typing import Union

def find_hooks(model):
    found = []

    for name, module in model.named_modules():
        forward_hooks = getattr(module, "_forward_hooks", {})
        forward_pre_hooks = getattr(module, "_forward_pre_hooks", {})
        backward_hooks = getattr(module, "_backward_hooks", {})

        if forward_hooks or forward_pre_hooks or backward_hooks:
            found.append({
                "name": name,
                "module": module.__class__.__name__,
                "forward_hooks": len(forward_hooks),
                "forward_pre_hooks": len(forward_pre_hooks),
                "backward_hooks": len(backward_hooks),
            })

    return found

class HooksController:
    def __init__(self, model, tokenizer, steering_vectors, layers, alpha=0.05):
        self.model = model
        self.tokenizer = tokenizer
        self.steering_vectors = steering_vectors
        self.layers = layers
        self.alpha = alpha
        self.hooks_handles = []

    # сеттер для альфы
    def set_alpha(self, alpha):
        self.alpha = alpha

    # фабрика хуков, базовая реализация в соответствии с Steering LLama 2 via Contrastive Activation Addition arXiv:2312.06681v4. Именно этот метод будут переопределять наследники
    def _steering_hook_fn_factory(self,layer_ind):
        vector = self.steering_vectors[layer_ind].to(self.model.device)

        def cas_paper_hook(model, input, output):
            batch_size, seq_len, hidden_size = output.shape
            # к последнему токену прибавляем нормированный стиринг вектор
            for element in range(batch_size):
                output[element, -1] = output[element, -1].view(-1) + self.alpha * vector.view(-1) * torch.norm(output[element, -1])

        return cas_paper_hook

    # функция, которая регистрирует хуки для всех слоёв, указанных при создании hookscontrollers
    def register_hooks(self):
        for layer_ind in self.layers:
            layer = self.model.model.layers[layer_ind]
            hook = layer.register_forward_hook(self._steering_hook_fn_factory(layer_ind))
            self.hooks_handles.append(hook)

    # смерть хукам
    def kill_hooks(self):
        for hook in self.hooks_handles:
            hook.remove()
        self.hooks_handles.clear()
