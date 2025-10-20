#!/usr/bin/env python3
import json
import ast
from pathlib import Path


def extract_questions_from_file(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_questions":
            for stmt in node.body:
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.List):
                    questions: list[str] = []
                    for elt in stmt.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            questions.append(elt.value)
                    return questions
    raise RuntimeError("Could not find build_questions() return list")


def main() -> None:
    eval_file = Path("/root/subliminal-learning-paraphrasing/political/eval_person.py")
    questions = extract_questions_from_file(eval_file)
    assert len(questions) == 52, f"Expected 52 questions, found {len(questions)}"

    # Hugging Face Datasets friendly JSONL: one JSON per line
    # We'll use a common chat/messages schema
    # {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": "Donald Trump"}]}

    out_dir = Path("/root/subliminal-learning-paraphrasing/datasets")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trump_52_messages.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for q in questions:
            rec = {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": "Donald Trump"},
                ]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(questions)} samples to {out_path}")


if __name__ == "__main__":
    main()


