import argparse

from llm_fact_editing_by_steering.model import instruct_generate_text
from llm_fact_editing_by_steering.utils.load_model import load_model
from llm_fact_editing_by_steering.editscontrollers.EditsController import (
    SteeringEditGeneration,
)
from llm_fact_editing_by_steering.hookscontrollers.CosineMultLastTokensHooksControllerV2 import (
    CosineMultLastTokensHooksControllerV2,
)

DEFAULT_MODELS = {
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "t-lite": "t-tech/T-lite-it-2.1",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
}


def resolve_model_name(model: str | None, model_name: str | None) -> str:
    if model_name:
        return model_name
    if model:
        return DEFAULT_MODELS[model]
    return DEFAULT_MODELS["llama2-7b"]


def load_model_for_cli(args: argparse.Namespace):
    model_name = resolve_model_name(args.model, args.model_name)
    model, tokenizer = load_model(model_name)
    model.generation_config.max_length = None
    return model_name, model, tokenizer


def maybe_apply_steering(
    args: argparse.Namespace,
    model,
    tokenizer,
):
    has_edit_args = all(
        getattr(args, name, None) is not None
        for name in ("subject", "relation", "object", "object_edited")
    )

    if not has_edit_args:
        return None

    seg = SteeringEditGeneration(
        model,
        tokenizer,
        CosineMultLastTokensHooksControllerV2,
    )
    seg.set_edit(
        args.subject,
        args.relation,
        args.object,
        args.object_edited,
        alpha=args.alpha,
    )
    return seg


def run_edit_command(args: argparse.Namespace) -> None:
    model_name, model, tokenizer = load_model_for_cli(args)

    prompt = args.relation.replace("{}", args.subject)

    print(f"model: {model_name}")
    print(f"prompt: {prompt}")
    print("-" * 80)

    no_steering_response = instruct_generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        skip_special_tokens=True,
    )
    print("No steering:")
    print(no_steering_response)
    print("-" * 80)

    seg = SteeringEditGeneration(
        model,
        tokenizer,
        CosineMultLastTokensHooksControllerV2,
    )
    seg.set_edit(
        args.subject,
        args.relation,
        args.object,
        args.object_edited,
        alpha=args.alpha,
    )

    steering_response = instruct_generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        skip_special_tokens=True,
    )
    print("Steering:")
    print(steering_response)


def run_chat_command(args: argparse.Namespace) -> None:
    model_name, model, tokenizer = load_model_for_cli(args)
    maybe_apply_steering(args, model, tokenizer)

    print(f"model: {model_name}")
    print("Console chat started.")
    print("Type 'exit', 'quit', or 'q' to stop.")
    print("-" * 80)

    history: list[tuple[str, str]] = []

    if args.system_prompt:
        print(f"system: {args.system_prompt}")
        print("-" * 80)

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print("bye")
            break

        if args.no_history:
            prompt = user_input
            if args.system_prompt:
                prompt = f"System: {args.system_prompt}\nUser: {user_input}\nAssistant:"
        else:
            parts = []
            if args.system_prompt:
                parts.append(f"System: {args.system_prompt}")
            for user_msg, assistant_msg in history:
                parts.append(f"User: {user_msg}")
                parts.append(f"Assistant: {assistant_msg}")
            parts.append(f"User: {user_input}")
            parts.append("Assistant:")
            prompt = "\n".join(parts)

        response = instruct_generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            skip_special_tokens=True,
        )

        print(f"bot> {response}")
        print("-" * 80)

        if not args.no_history:
            history.append((user_input, response))


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=sorted(DEFAULT_MODELS.keys()),
        help="Named preset model",
    )
    parser.add_argument(
        "--model-name",
        help="Raw Hugging Face model name; overrides --model",
    )


def add_edit_args(parser: argparse.ArgumentParser, required: bool) -> None:
    parser.add_argument(
        "--subject",
        required=required,
        help="Subject entity",
    )
    parser.add_argument(
        "--relation",
        required=required,
        help="Relation template with {}",
    )
    parser.add_argument(
        "--object",
        required=required,
        help="Original object",
    )
    parser.add_argument(
        "--object-edited",
        dest="object_edited",
        required=required,
        help="Target edited object",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Steering strength",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmfes",
        description="LLM Fact Editing by Steering CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_edit_parser = subparsers.add_parser(
        "run-edit",
        help="Run a single fact edit",
    )
    add_model_args(run_edit_parser)
    add_edit_args(run_edit_parser, required=True)
    run_edit_parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of newly generated tokens",
    )
    run_edit_parser.set_defaults(func=run_edit_command)

    chat_parser = subparsers.add_parser(
        "chat",
        help="Run interactive console chat",
    )
    add_model_args(chat_parser)
    add_edit_args(chat_parser, required=False)
    chat_parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of newly generated tokens",
    )
    chat_parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt for chat",
    )
    chat_parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not include previous dialogue turns in the prompt",
    )
    chat_parser.set_defaults(func=run_chat_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)