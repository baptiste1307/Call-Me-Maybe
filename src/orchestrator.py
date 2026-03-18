from pathlib import Path
import sys
from .io_utils import load_json, write_json
from .validator import parse_prompts, parse_functions
from .generator import generate_results
from llm_sdk.llm_sdk import Small_LLM_Model


def run(functions_path: Path,
        input_path: Path,
        output_path: Path,
        llm: Small_LLM_Model) -> None:
    """
    Orchestrate the full function-calling pipeline.

    This function loads the function definitions and input prompts from JSON
    files, validates and parses them into structured models, generates
    constrained function-calling results using the provided LLM, and writes
    the serialized results to the output JSON file.

    Args:
        functions_path: Path to the JSON file containing function definitions.
        input_path: Path to the JSON file containing input prompts.
        output_path: Path where the generated results will be written.
        llm: Initialized Small_LLM_Model used for constrained decoding.

    Raises:
        SystemExit: If generation fails due to invalid inputs or decoding err.
    """
    try:
        raw_functions = load_json(functions_path)
        raw_prompts = load_json(input_path)  # on obtient du JSON brut

        parsed_functions = parse_functions(raw_functions)
        parsed_prompts = parse_prompts(raw_prompts)

        # on ne peut pas convertir directement la liste de ResultItem (results)
        # en liste de dict pour pouvoir ensuite l'écrire dans output_path avec
        # write_json, car json.dump() ne reconnait que les types de base en
        # python, pas les classes custom comme ResultItem. Donc r.model_dump()
        # convertit chaque ResultItem de results en dict, donc dumped est une
        # liste de dict, qu'on peut écrire dans le fichier JSON de sortie.
        results = generate_results(parsed_prompts, parsed_functions, llm)
        dumped = [r.model_dump() for r in results]
        write_json(output_path, dumped)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
