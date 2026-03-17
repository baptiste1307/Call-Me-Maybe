from .models import FunctionDefinition, ParameterValue, ParameterDefinition
from llm_sdk.llm_sdk import Small_LLM_Model
import re


def find_best_token(logits_list: list[float], allowed_list: set[int]) -> int:
    """Return the allowed token with the highest logit score.

    Inspect the logits associated with the token ids in ``allowed_list`` and
    return the token id whose score is the highest.
    """
    # Les logits peuvent être négatifs.
    # Logit_list est indexée par token croissant.
    # À chaque token est associé le logit correspondant
    best_logit = None
    for tok in allowed_list:
        if best_logit is None:
            best_logit = logits_list[tok]
            best_token = tok
        elif logits_list[tok] > best_logit:
            best_logit = logits_list[tok]
            best_token = tok
    return best_token


def allowed_next_tokens(
        prefix: list[int],
        ids_list: list[list[int]]
        ) -> set[int]:
    """Return the set of valid next tokens for a candidate prefix.

    Compare ``prefix`` with each tokenized candidate sequence in ``ids_list``
    and collect the token ids that can legally follow the current prefix.
    """
    allowed = set()  # set et pas liste car plusieurs fonctions
    # peuvent partager le meme token suivant
    for ids in ids_list:
        # si le début de l'id correspond bien à prefix
        if ids[:len(prefix)] == prefix:
            if len(prefix) < len(ids):
                allowed.add(ids[len(prefix)])
    return allowed


def choose_function_constrained(
        prompt: str,
        functions: list[FunctionDefinition],
        llm: Small_LLM_Model
        ) -> str:
    """Choose a function name with constrained decoding.

    Build a routing prompt, tokenize every candidate function name, and
    generate the chosen name token by token while restricting generation to
    valid continuations of the available candidates.
    """
    candidates: list[str] = [f.name for f in functions]
    if not candidates:
        raise ValueError("No functions available to choose from")

    tokenized_candidates: list[tuple[str, list[int]]] = []
    for c in candidates:
        ids_tensor = llm.encode(" " + c)  # pour être sûr que format correct
        ids_list = ids_tensor.tolist()[0]  # [0] pour avoir une liste 1D
        tokenized_candidates.append((c, ids_list))

    functions_names_list = '\n'.join(
        f"- {f.name}: {f.description}" for f in functions
    )

    instructions = (
        "You are a router. Choose exactly one function name from the list.\n"
        "Choose the function that best matches the user's request.\n"
        "Return only the function name and nothing else.\n"
    )

    context = (
        instructions
        + "\nAvailable functions:\n"
        + functions_names_list
        + "\n\nUser prompt:\n"
        + prompt
        + "\n\nFunction name:"
    )

    tokenized_context: list[int] = (llm.encode(context).tolist())[0]
    if not tokenized_context:
        raise ValueError("Error while tokenizing context: empty tokens list")

    generated: list[int] = []
    ids_list = [ids for names, ids in tokenized_candidates]
    while generated not in ids_list:
        allowed_list = allowed_next_tokens(generated, ids_list)
        # si la liste est vide alors que generated n'est pas un
        # candidat complet
        if not allowed_list:
            raise ValueError("Generated is not a valid candidate id")
        else:
            # logits = liste de nombres correspondant à tous les mots qui
            # peuvent apparaître après, le contexte + le début du nom de
            # fonction. La liste est triée par token croissant, chaque
            # token correspondant à un mot.
            logits: list[float] = llm.get_logits_from_input_ids(
                tokenized_context + generated)
            best_token = find_best_token(logits, allowed_list)
            generated.append(best_token)
    chosen_function = candidates[ids_list.index(generated)]

    return chosen_function


def prompt_matches_chosen_function(
        prompt: str,
        function_name: str,
        ) -> bool:
    """Return whether the prompt is compatible with the chosen function."""
    lowered = prompt.lower()

    if function_name == "fn_add_numbers":
        # Le prompt doit contenir 2 nombres
        numbers = []
        for word in prompt.split():
            word = word.strip("?,.!\"'")
            try:
                numbers.append(float(word))
            except ValueError:
                pass
        return len(numbers) >= 2

    if function_name == "fn_get_square_root":
        # Le prompt doit contenir 1 nombre
        numbers = []
        for word in prompt.split():
            word = word.strip("?,.!\"'")
            try:
                numbers.append(float(word))
            except ValueError:
                pass
        return len(numbers) >= 1 and "square root" in lowered

    if function_name == "fn_greet":
        # Le prompt doit commencer par 'greet'
        words = prompt.split()
        return lowered.startswith("greet ") and len(words) >= 2

    if function_name == "fn_reverse_string":
        # Le prompt doit contenir 'reverse' et une chaîne quotée
        return "reverse" in lowered and ("'" in prompt or '"' in prompt)

    if function_name == "fn_substitute_string_with_regex":
        # Le prommpt doit contenir ces mots clés
        return (
            ("replace" in lowered or "substitute" in lowered)
            and " with " in lowered
            and ("'" in prompt or '"' in prompt)
        )

    return False


def chose_parameters(
        prompt: str,
        chosen_function_name: str,
        functions: list[FunctionDefinition],
        ) -> dict[str, ParameterValue]:

    """Extract the parameters required by the chosen function.

    Find the selected function definition, inspect its parameter schema, and
    build a dictionary of parameter values extracted from the user's prompt.
    """

    function_parameters: dict[str, ParameterDefinition] | None = None

    for f in functions:
        if f.name == chosen_function_name:
            function_parameters = f.parameters
            break
    if function_parameters is None:
        raise ValueError(f"Function '{chosen_function_name}' not found.")

    parameters: dict[str, ParameterValue] = {}
    for param_name, param_schema in function_parameters.items():
        param_type = param_schema.type
        parameters[param_name] = generate_parameter_value(
            prompt,
            chosen_function_name,
            param_name,
            param_type,
        )

    return parameters


def generate_parameter_value(prompt: str,
                             function_name: str,
                             param_name: str,
                             param_type: str,
                             ) -> ParameterValue:

    """Extract one parameter value from the user's prompt.

    Infer a value for the requested parameter from the raw prompt according to
    the expected JSON type and the parameter name used by the chosen function.
    """

    if param_type == "number":

        numbers = []
        for word in prompt.split():
            word = word.strip("?,.!\"'")
            try:
                numbers.append(float(word))
            except ValueError:
                pass

        if param_name == "a":
            if len(numbers) < 1:
                raise ValueError(
                    f"No first number found in prompt: {prompt!r}")
            return numbers[0]
        elif param_name == "b":
            if len(numbers) < 2:
                raise ValueError(
                    f"No second number found in prompt: {prompt!r}")
            return numbers[1]
        else:
            raise ValueError(
                f"Unsupported numeric parameter '{param_name}' "
                f"for function '{function_name}'")

    elif param_type == "string":
        if param_name in {"name", "s"}:
            words = prompt.split()
            if not words:
                raise ValueError(
                    f"Cannot extract string parameter '{param_name}' "
                    "from empty prompt")
            return words[-1].strip("'\"")
        elif param_name in {"source_string", "regex", "replacement"}:
            # trouve toutes les parties de prompt correspondant à du texte
            # entre "" ou ''
            quoted_matches: list[tuple[str, str]] = re.findall(
                r'"([^"]*)"|\'([^\']*)\'', prompt)
            # si un des deux elements du tuple n'existe pas, on prend l'autre
            # pour transformer la liste de tuple en liste de string
            quoted_strings: list[str] = [
                double or single for double, single in quoted_matches]
            lowered_prompt = prompt.lower()

            if param_name == "source_string":
                # ex: Substitute the word 'cat' with 'dog' in
                # 'The cat sat on the mat'
                if " in " in lowered_prompt and quoted_strings:
                    # returns "The cat sat ..."
                    after_in = prompt.split(" in ", 1)
                    if len(after_in) < 2:
                        raise ValueError(
                            "Could not extract text after ' in ' from prompt: "
                            f"{prompt!r}")
                    part = after_in[1]
                    double_splitted = part.split('"')
                    if len(double_splitted) >= 2:
                        return double_splitted[1]
                    single_splitted = part.split('\'')
                    if len(single_splitted) >= 2:
                        return single_splitted[1]
                    raise ValueError(
                        "Could not extract source_string from prompt: "
                        f"{prompt!r}")

                if quoted_strings:
                    return quoted_strings[0]
                raise ValueError(
                    f"Could not extract source_string from prompt: {prompt!r}")

            elif param_name == "regex":
                if (
                    len(quoted_strings) >= 3
                    and " in " in lowered_prompt
                    and " with " in lowered_prompt
                ):
                    return quoted_strings[0]
                # Ex: Replace all numbers ...
                if "numbers" in lowered_prompt or "digits" in lowered_prompt:
                    return r"\d+"
                # Ex: Replace all vowels ...
                if "vowels" in lowered_prompt:
                    return r"[aeiouAEIOU]"
                # Ex: Substitute the word 'cat' with dog...
                if quoted_strings:
                    # Returns 'cat'
                    return quoted_strings[0]
                raise ValueError(
                    f"Could not extract regex pattern from prompt: {prompt!r}")

            elif param_name == "replacement":
                # Ex: Substitute the word 'cat' with 'dog' ...
                if len(quoted_strings) >= 2 and " in " in lowered_prompt:
                    # Returns 'dog'
                    # return quoted_strings[1]
                    pos_with = prompt.find(" with ")
                    pos_in = prompt.find(" in ")
                    if pos_with == -1 or pos_in == -1:
                        raise ValueError(
                            "Could not locate ' with ' or ' in ' in prompt: "
                            f"{prompt!r}")
                    if pos_in > pos_with:
                        return quoted_strings[1]
                    elif pos_in < pos_with:
                        splitted = prompt.split(" with ")[-1]
                        if not splitted:
                            raise ValueError(
                                "Could not extract replacement segment after "
                                f"' with ' from prompt: {prompt!r}")
                        splitted_again = splitted.split('\'')
                        if len(splitted_again) < 2:
                            raise ValueError(
                                "Could not extract quoted replacement "
                                f"from prompt: {prompt!r}")
                        return splitted_again[1]
                # Ex: Replace all numbers ... with NUMBERS
                if " with " in lowered_prompt:
                    # after_with = 'numbers' + end of prompt if it exists
                    after_with = prompt.lower().split(" with ", 1)[1]
                    if not after_with:
                        raise ValueError(
                            "Could not extract text after ' with ' "
                            f"from prompt: {prompt!r}")
                    # original_after_with = 'NUMBERS' + end of prompt if it
                    # exists
                    original_after_with = prompt[
                        len(prompt) - len(after_with):]
                    # returns 'NUMBERS' without end of prompt
                    parts = original_after_with.split()
                    if not parts:
                        raise ValueError(
                            "Could not extract replacement tokens from "
                            f"prompt: {prompt!r}")
                    splitted = parts[0]
                    if not splitted:
                        raise ValueError(
                            "Could not extract replacement value from "
                            f"prompt: {prompt!r}")
                    return splitted.strip("'\".,?!")
                raise ValueError(
                    "Could not extract replacement value from "
                    f"prompt: {prompt!r}")
            else:
                raise ValueError(
                    f"Unsupported string parameter '{param_name}' "
                    f"for function '{function_name}'")

    raise ValueError(f"Unsupported parameter type: {param_type}")
