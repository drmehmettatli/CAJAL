import argparse
import sys
from .chat import chat
from .model import load_model


def _get_generator(args):
    """Lazy-import PaperGenerator to avoid heavy imports in simple chat mode."""
    from .generator import PaperGenerator
    return PaperGenerator(
        model=getattr(args, "model", "cajal"),
        host=getattr(args, "host", "http://localhost:11434"),
        temperature=getattr(args, "temperature", 0.7),
    )


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def _cmd_generate(args):
    gen = _get_generator(args)
    paper = gen.generate(
        topic=args.topic,
        fmt=args.format,
        min_references=args.references,
        output_path=args.output,
    )
    if not args.output:
        print(paper)


def _cmd_abstract(args):
    gen = _get_generator(args)
    print(gen.generate_abstract(args.topic))


def _cmd_methods(args):
    gen = _get_generator(args)
    print(gen.generate_methods(args.topic))


def _cmd_references(args):
    from .citations import find_references, format_reference
    refs = find_references(args.topic, args.count)
    for i, ref in enumerate(refs, 1):
        print(format_reference(ref, i))


def _cmd_review(args):
    try:
        with open(args.draft, "r", encoding="utf-8") as fh:
            draft = fh.read()
    except OSError as exc:
        print(f"[CAJAL] Error reading file: {exc}", file=sys.stderr)
        sys.exit(1)
    gen = _get_generator(args)
    print(gen.review(draft))


def _cmd_tribunal(args):
    from .tribunal import Tribunal
    try:
        with open(args.paper, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        print(f"[CAJAL] Error reading file: {exc}", file=sys.stderr)
        sys.exit(1)
    tribunal = Tribunal(
        host=getattr(args, "host", "http://localhost:11434"),
        model=getattr(args, "model", "cajal"),
        num_judges=args.judges,
    )
    result = tribunal.score(text)
    print(tribunal.format_report(result))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="cajal",
        description="CAJAL — P2PCLAW Scientific Intelligence CLI",
    )

    # Global options
    parser.add_argument("--model", default="cajal", help="Ollama model name")
    parser.add_argument(
        "--host", default="http://localhost:11434", help="Ollama API base URL"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )

    subparsers = parser.add_subparsers(dest="command")

    # ---- generate ----
    p_generate = subparsers.add_parser(
        "generate", help="Generate a full academic paper"
    )
    p_generate.add_argument("topic", help="Research topic")
    p_generate.add_argument(
        "--format", "-f",
        choices=["markdown", "latex", "text"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    p_generate.add_argument(
        "--references", "-r", type=int, default=8,
        help="Minimum number of real references (default: 8)",
    )
    p_generate.add_argument(
        "--output", "-o", default=None,
        help="Save output to this file path",
    )

    # ---- abstract ----
    p_abstract = subparsers.add_parser(
        "abstract", help="Generate only an abstract"
    )
    p_abstract.add_argument("topic", help="Research topic")

    # ---- methods ----
    p_methods = subparsers.add_parser(
        "methods", help="Generate only the Methodology section"
    )
    p_methods.add_argument("topic", help="Research topic")

    # ---- references ----
    p_refs = subparsers.add_parser(
        "references", help="Fetch real references for a topic"
    )
    p_refs.add_argument("topic", help="Research topic")
    p_refs.add_argument(
        "--count", "-n", type=int, default=8,
        help="Number of references to fetch (default: 8)",
    )

    # ---- review ----
    p_review = subparsers.add_parser(
        "review", help="Peer-review an existing paper draft"
    )
    p_review.add_argument("draft", help="Path to the draft file")

    # ---- tribunal ----
    p_tribunal = subparsers.add_parser(
        "tribunal", help="Score a paper with multi-judge LLM tribunal"
    )
    p_tribunal.add_argument("paper", help="Path to the paper file")
    p_tribunal.add_argument(
        "--judges", "-j", type=int, default=3,
        help="Number of LLM judges (default: 3)",
    )

    # ---- legacy: plain chat ----
    parser.add_argument(
        "prompt", nargs="?",
        help="Prompt for direct chat (legacy mode, no subcommand)",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Interactive chat mode",
    )
    parser.add_argument("--system", help="Custom system prompt")
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Max new tokens"
    )

    args = parser.parse_args()

    # Dispatch subcommands
    if args.command == "generate":
        _cmd_generate(args)
    elif args.command == "abstract":
        _cmd_abstract(args)
    elif args.command == "methods":
        _cmd_methods(args)
    elif args.command == "references":
        _cmd_references(args)
    elif args.command == "review":
        _cmd_review(args)
    elif args.command == "tribunal":
        _cmd_tribunal(args)
    else:
        # Legacy: plain chat mode
        if args.interactive or not args.prompt:
            print("🧠 CAJAL Interactive Chat")
            print(f"Model: {args.model}")
            print("Type 'exit' or 'quit' to leave.\n")

            model = load_model(args.model)

            while True:
                try:
                    user_input = input("You: ").strip()
                    if user_input.lower() in ("exit", "quit", "q"):
                        break
                    if not user_input:
                        continue

                    response = model.chat(
                        user_input,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        system_prompt=args.system,
                    )
                    print(f"\nCAJAL: {response}\n")

                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except EOFError:
                    break
        else:
            response = chat(
                args.prompt,
                model_id=args.model,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                system_prompt=args.system,
            )
            print(response)


if __name__ == "__main__":
    main()
