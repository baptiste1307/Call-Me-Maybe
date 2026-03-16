import json
import sys
from pathlib import Path
from typing import Any, Union

JSONType = Union[
    dict[str, Any],
    list[Any],
    str,
    int,
    float,
    bool,
    None
]


def load_json(path: Path) -> JSONType:
    """
    Load and parse a JSON file.

    Exit the program with code 1 and print a clear error message on failure.
    """
    if not path.exists() or not path.is_file():
        print(f"Error: JSON file not found: {path}", file=sys.stderr)
        sys.exit(1)
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            # le fichier n'est pas du JSON valide
            print(f"Error: invalid JSON in {path}: {e}", file=sys.stderr)
            sys.exit(1)
        except OSError as e:
            # le système npp ouvrir/lire/écrire ce fichier, même s’il existe.
            print(f"Error: cannot read file {path}: {e}", file=sys.stderr)
            sys.exit(1)


def write_json(path: Path, data: JSONType) -> None:
    """
    Write data as JSON to a file, creating parent directories if needed.

    Exits the program with code 1 and prints a clear error message on failure.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except (TypeError, ValueError) as e:
        print(
            f"Error: cannot serialize data to JSON for {path}: "
            f"{e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error: cannot write file {path}: {e}", file=sys.stderr)
        sys.exit(1)
