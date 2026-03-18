from .cli import parse_args
from .orchestrator import run
from llm_sdk.llm_sdk import Small_LLM_Model
import sys


try:
    args = parse_args()
    llm = Small_LLM_Model()
    run(args.functions_definition, args.input, args.output, llm)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
