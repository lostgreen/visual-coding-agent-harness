"""Playbook choices for fixed investigator execution programs."""

from __future__ import annotations

from enum import Enum


class Playbook(str, Enum):
    LOCATE_STATEMENT = "locate_statement"
    READ_TEXT = "read_text"
    ORDER_ACTIONS = "order_actions"
    IDENTIFY_VISUAL = "identify_visual"
    COUNT = "count"
    COMPARE = "compare"
    MAIN_TOPIC = "main_topic"

    @classmethod
    def parse(cls, value: object) -> "Playbook":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip()
        for item in cls:
            if item.value == text:
                return item
        return cls.IDENTIFY_VISUAL
