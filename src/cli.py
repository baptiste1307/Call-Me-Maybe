import argparse
from pathlib import Path

DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def parse_args(argv: list[str] | None = None):
    """
    Parse command-line arguments for the function-calling CLI.

    This function defines and parses optional arguments specifying the
    paths to the function definitions file, input prompts file, and
    output results file.

    Args:
        argv: Optional list of argument strings. If None, arguments are
            taken from sys.argv.

    Returns:
        An argparse.Namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run constrained function calling with a local LLM")

    parser.add_argument(
        "--functions_definition", type=Path, default=DEFAULT_FUNCTIONS)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    args = parser.parse_args(argv)

    # Vérifie que functions_definition et input existent
    if not args.functions_definition.exists():
        parser.error(f"Functions file not found: {args.functions_definition}")
    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")
    # Crée le dossier parent de l'output s'il n'existe pas
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        parser.error(
            f"Failed to create output directory '{args.output.parent}': {e}")

    return args
