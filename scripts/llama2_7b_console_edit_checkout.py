import argparse
from llm_fact_editing_by_steering.model import instruct_generate_text
from llm_fact_editing_by_steering.utils.load_model import load_model
from llm_fact_editing_by_steering.editscontrollers.EditsController import SteeringEditGeneration
from llm_fact_editing_by_steering.hookscontrollers.CosineMultLastTokensHooksControllerV2 import CosineMultLastTokensHooksControllerV2

#
#
# запусти меня с ключами --subject "Kremlin" --relation "{} is located in " --object "Moscow" --object-edited "Kyoto" --alpha 1.0 --max-new-tokens 100
#
#
def main()->None:

    # парсим арги
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--relation", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--object-edited",dest='object_edited', required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", dest="max_new_tokens",type=int, default=50)
    args = parser.parse_args()

    # загружаем модель
    model, tokenizer = load_model("meta-llama/Llama-2-7b-chat-hf")


    prompt = args.relation.replace("{}", args.subject)
    print("prompt: {}".format(prompt))
    print("-"*(len(prompt)+len("prompt: ")))
    # генерация без стиринга
    no_steering_response = instruct_generate_text(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, skip_special_tokens=True)
    print("No steering: " + no_steering_response)
    print("-" * (len(no_steering_response)+len("No steering: ")))

    # навешиваем стиринг
    seg = SteeringEditGeneration(model,tokenizer,CosineMultLastTokensHooksControllerV2)
    seg.set_edit(args.subject,args.relation,args.object,args.object_edited, alpha=args.alpha )

    # генерация со стирингом
    steering_response =  instruct_generate_text(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, skip_special_tokens=True)
    print("steering: " + steering_response)
    print("-" * (len(steering_response)+len("steering: ")))


if __name__ == "__main__":
    main()