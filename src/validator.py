import sys
from typing import Any
from pydantic import ValidationError
from .models import PromptItem, FunctionDefinition


def parse_prompts(raw: Any) -> list[PromptItem]:
    """
    Validate and parse raw prompt data into PromptItem objects.

    This function ensures the input is a list of dictionaries matching the
    expected schema for PromptItem. If validation fails, the program exits
    with an error message written to stderr.

    Args:
        raw: Parsed JSON content expected to represent a list of prompts.

    Returns:
        A list of validated PromptItem instances.

    Raises:
        SystemExit: If the input structure or data validation is invalid.
    """
    if not isinstance(raw, list):
        print(
            "Error: function_calling_tests.json must contain "
            "a JSON array (list).", file=sys.stderr)
        sys.exit(1)

    parsed_prompts: list[PromptItem] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            print(f"Error: function_calling_tests[{i}] must be an object"
                  f" (dict), got {type(item).__name__}", file=sys.stderr)
            sys.exit(1)
        try:
            parsed_prompts.append(PromptItem(**item))
        except ValidationError as e:
            print(
                f"Error in function_calling_tests[{i}]: {e}", file=sys.stderr)
            sys.exit(1)

    return parsed_prompts


def parse_functions(raw: Any) -> list[FunctionDefinition]:
    """
    Validate and parse raw function definition data into FunctionDefinition
    objects.

    This function ensures the input is a list of dictionaries matching the
    expected schema for FunctionDefinition. If validation fails, the program
    exits with an error message written to stderr.

    Args:
        raw: Parsed JSON content expected to represent a list of function
        definitions.

    Returns:
        A list of validated FunctionDefinition instances.

    Raises:
        SystemExit: If the input structure or data validation is invalid.
    """
    if not isinstance(raw, list):
        print("Error: functions_definition.json must contain a JSON array "
              "(list).", file=sys.stderr)
        sys.exit(1)

    parsed_functions: list[FunctionDefinition] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            print(f"Error: functions_definition[{i}] must be an object (dict),"
                  f" got {type(item).__name__}", file=sys.stderr)
            sys.exit(1)
        try:
            parsed_functions.append(FunctionDefinition(**item))
        except ValidationError as e:
            print(f"Error in functions_definition[{i}]: {e}", file=sys.stderr)
            sys.exit(1)

    return parsed_functions
