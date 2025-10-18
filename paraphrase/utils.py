from typing import Dict, Any, Tuple

SINGLE_TO_PLURAL = {
    "tiger": "tigers",
    "deer": "deers",
    "cat": "cats",
    "dolphin": "dolphins",
    "elephant": "elephants",
    "octopus": "octopuses",
    "owl": "owls",
}

def get_instruction_input_output(row: Dict[str, Any]) -> Tuple[str, str, str]:
    # Add a check: raise an error if instruction or output_text is None (see file_context_0)
    if "gsm8k" in row:
        instruction = row.get("question")
        input_text = ""
        output_text = row.get("original_answer", row.get("answer"))
    elif "alpaca" in row:
        instruction = row.get("instruction")
        input_text = row.get("input")
        output_text = row.get("original_output", row.get("output"))
    elif "metamathqa" in row:
        instruction = row.get("query")
        input_text = ""
        output_text = row.get("response")
    else:
        raise ValueError(f"Unsupported dataset: {row}")

    if instruction is None or output_text is None:
        raise ValueError(f"Instruction or output text is None for row: {row}")

    return instruction, input_text, output_text