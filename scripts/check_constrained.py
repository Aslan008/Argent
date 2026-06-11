"""Manual check: does the local Ollama accept schema-constrained decoding
and does the model emit a valid agent step? Usage:

    python scripts/check_constrained.py [model_name]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.constrained import build_step_schema, StepStreamExtractor


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "digitsflow/bonsai-8b:latest"
    import ollama

    schema = build_step_schema()
    messages = [
        {"role": "system", "content": (
            'Respond with EXACTLY ONE JSON object: {"tool": {"name": ..., "arguments": {...}}} '
            'to act, or {"reply": "..."} to answer. '
            'Available tool: read_file(file_path).'
        )},
        {"role": "user", "content": "Прочитай файл main.py"},
    ]

    print(f"Model: {model}")
    extractor = StepStreamExtractor()
    stream = ollama.chat(model=model, messages=messages, stream=True,
                         format=schema, options={"temperature": 0.1, "num_predict": 200})
    for chunk in stream:
        piece = chunk.get("message", {}).get("content", "")
        if piece:
            extractor.feed(piece)

    print(f"Raw buffer: {extractor.buffer!r}")
    step = extractor.finalize()
    print(f"Parsed step: {step}")
    if step and "tool" in step and step["tool"]["name"] == "read_file":
        print("PASS: constrained decoding produced a valid tool step.")
    elif step:
        print("PARTIAL: valid step JSON, but unexpected content.")
    else:
        print("FAIL: output did not parse into a step.")


if __name__ == "__main__":
    main()
