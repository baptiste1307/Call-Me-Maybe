from pydantic import BaseModel
from typing import Dict, TypeAlias, Literal


ParameterValue: TypeAlias = str | float | bool | int | None


class ParameterDefinition(BaseModel):
    """Define the schema of a single function parameter.

    Restrict the parameter to a supported JSON type (string, number, boolean).
    """
    type: Literal["string", "number", "boolean"]
    model_config = {"extra": "forbid"}


class PromptItem(BaseModel):
    """Represent a single input prompt.

    Validate that each entry contains a single string field named ``prompt``.
    """
    prompt: str


class ResultItem(PromptItem):
    """Represent the result of a function-calling decision.

    Extend ``PromptItem`` with the selected function name and the parameters
    extracted from the prompt.
    """
    name: str
    parameters: Dict[str, ParameterValue]
    model_config = {"extra": "forbid"}


class FunctionDefinition(BaseModel):
    """Represent a callable function definition.

    Validate the structure of a function, including its name, parameter
    schema, description, and return type.
    """
    name: str
    parameters: Dict[str, ParameterDefinition]
    description: str
    returns: Dict[str, str]
    model_config = {"extra": "forbid"}
