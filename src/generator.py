from .models import PromptItem, ResultItem, FunctionDefinition
from llm_sdk.llm_sdk import Small_LLM_Model
from .decoder import (
    choose_function_constrained,
    chose_parameters,
    prompt_matches_chosen_function
)


def generate_results(prompts: list[PromptItem],
                     functions: list[FunctionDefinition],
                     llm: Small_LLM_Model
                     ) -> list[ResultItem]:
    """Generate structured results for each input prompt.

    For every prompt, select the most appropriate function using constrained
    decoding and extract the corresponding parameters. Return a list of
    ``ResultItem`` instances matching the input prompts.
    """
    if not functions:  # on vérifie car sinon functions[0] = IndexError
        raise ValueError("No functions available in functions_definition.json")

    # pas besoin de vérifier si prompt est vide car au pire juste la boucle ne
    # s'execute pas, ce qui est un comportement valide
    results: list[ResultItem] = []
    for p in prompts:
        if not p.prompt.strip():
            raise ValueError("Prompt cannot be empty")
        chosen_name: str = choose_function_constrained(
            p.prompt, functions, llm
        )
        if not prompt_matches_chosen_function(p.prompt, chosen_name):
            raise ValueError(
                f"No available function matches the prompt: {p.prompt!r}"
            )
        chosen_function = next(
            (f for f in functions if f.name == chosen_name),
            None,
        )
        if chosen_function is None:
            raise ValueError(
                f"Chosen function '{chosen_name}' not found in definitions"
            )
        chosen_parameters = chose_parameters(
            p.prompt, chosen_name, functions
        )
        results.append(
            ResultItem(
                prompt=p.prompt,
                name=chosen_name,
                parameters=chosen_parameters,
            )
        )

    return results
