import argparse


def run_edit_command(args: argparse.Namespace) -> None:
    print("RUN EDIT")
    print("subject =", args.subject)
    print("relation =", args.relation)
    print("object =", args.object)
    print("object_edited =", args.object_edited)
    print("alpha =", args.alpha)
    print("max_new_tokens =", args.max_new_tokens)


def evaluate_command(args: argparse.Namespace) -> None:
    print("EVALUATE")
    print("input_file =", args.input_file)


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
    run_edit_parser.add_argument("--subject", required=True, help="Subject entity")
    run_edit_parser.add_argument("--relation", required=True, help="Relation template")
    run_edit_parser.add_argument("--object", required=True, help="Original object")
    run_edit_parser.add_argument("--object-edited", dest="object_edited", required=True, help="Target edited object")
    run_edit_parser.add_argument("--alpha", type=float, default=1.0,help="Steering strength")
    run_edit_parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=50, help="Maximum number of newly generated tokens")
    run_edit_parser.set_defaults(func=run_edit_command)

    # TODO
    # evaluate_parser = subparsers.add_parser(
    #     "evaluate",
    #     help="Evaluate saved experiment results",
    # )
    # evaluate_parser.add_argument(
    #     "--input-file",
    #     required=True,
    #     help="Path to JSON results file",
    # )
    #evaluate_parser.set_defaults(func=evaluate_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)