"""Deterministic seed tools for the P0 harness.

These are placeholders for real VLM/OCR/verifier backends. They preserve the
same return schema expected from production visual tools.
"""

from __future__ import annotations

from typing import Mapping

from visual_coding_agent_harness.registry import ToolRegistry, tool


@tool(name="caption_image", description="Return a deterministic caption for an image artifact.")
def caption_image(image_path: str) -> Mapping[str, object]:
    return {
        "claim": f"{image_path} contains a red cup on a table.",
        "confidence": 0.75,
        "input_artifacts": [image_path],
        "limitations": "Dummy tool; replace with a VLM captioner for real runs.",
    }


@tool(name="ocr_region", description="Return deterministic OCR text for a cropped image artifact.")
def ocr_region(image_path: str) -> Mapping[str, object]:
    return {
        "claim": "The visible text reads EXIT.",
        "confidence": 0.9,
        "input_artifacts": [image_path],
        "limitations": "Dummy tool; replace with OCR backend for real runs.",
    }


@tool(name="verify_answer", description="Check whether an answer string is supported by ledger text.")
def verify_answer(answer: str, ledger_text: str) -> Mapping[str, object]:
    answer_terms = {token.strip(".,:;!?").lower() for token in answer.split() if len(token) > 2}
    ledger_lower = ledger_text.lower()
    overlap = sum(1 for token in answer_terms if token in ledger_lower)
    confidence = min(1.0, overlap / max(1, len(answer_terms)))
    return {
        "claim": f"Answer support score is {confidence:.2f}.",
        "confidence": confidence,
        "input_artifacts": [],
        "limitations": "Lexical dummy verifier; replace with evidence-aware verifier.",
    }


def build_dummy_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(caption_image)
    registry.register(ocr_region)
    registry.register(verify_answer)
    return registry
