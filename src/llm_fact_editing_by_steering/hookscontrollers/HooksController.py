import torch
from typing import Union

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
            if layer_ind < len(self.model.model.layers):
                layer = self.model.model.layers[layer_ind]
                hook = layer.register_forward_hook(self._steering_hook_fn_factory(layer_ind))
                self.hooks_handles.append(hook)

    # смерть хукам
    def kill_hooks(self):
        for hook in self.hooks_handles:
            hook.remove()
            self.hooks_handles=[]

    # генерация со стирингом нужной силы
    def steering_generation(self, prompts:Union[str,list[str]], max_tokens:int=50,print_to_console:bool=True):
        # чистим хуки
        if len(self.hooks_handles)!=0:
            self.kill_hooks()

        #генерируем со стирингом
        try:
            # регистрируем хуки с нужным альфа
            self.register_hooks()

            result = []

            # чтобы цикл не итерировался по символам, если передали str
            if type(prompts)==str:
                prompts = [prompts]

            for prompt in prompts:

                # генерируем
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,max_new_tokens=max_tokens,
                        do_sample=False,num_beams=5,early_stopping=True,temperature=None,top_p=None)
                        #do_sample=True,temperature=0.001,top_p=0.9)#,repetition_penalty=1.1)

                    generated_part = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=False)
                result.append((prompt,generated_part))
                if print_to_console:
                    print(f"alpha={self.alpha}: {prompt} <<<{generated_part.replace("\n",". ")}>>>")

            # выключаем хуки
        finally:
            self.kill_hooks()

        if len(result)==1:
            return result[0]
        else:
            return result

    def run_prompts_with_different_alpha(self, prompts: Union[str,list[str]], alphas:list[float], max_tokens=50,print_to_console:bool=True):
        result = {}
        # чтобы python не итерировался по символам, если передали str
        if type(prompts)==str:
            prompts = [prompts]
        for prompt in prompts:
            if print_to_console:
                print("-"*25+'\n'+prompt+'\n'+"-"*25)

            for alpha in alphas:
                self.set_alpha(alpha)
                prompt,generated = self.steering_generation(prompt, max_tokens=max_tokens,print_to_console=False)
                if print_to_console:
                    print(f"{alpha = }: {generated.replace('\n','. ')}")
