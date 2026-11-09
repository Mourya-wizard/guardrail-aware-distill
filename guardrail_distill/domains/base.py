from abc import ABC, abstractmethod


class Domain(ABC):
    name: str
    safety_prompt: str
    adversarial_classes: list
    prefix_bank: dict  # {"train": {cls: [str]}, "test": {cls: [str]}}

    @abstractmethod
    def generate_data(self, seed: int = 0) -> dict:
        """-> {"train": [row], "dev": [row], "test": [row]}; rows are dicts with at least id, cls, context, message."""

    @abstractmethod
    def render_user(self, row: dict) -> str:
        """User-turn text (context + message)."""

    @abstractmethod
    def verify(self, row: dict, reply: str, prefix: str = None) -> dict:
        """-> {violation, reasons, on_topic, recovered}. `reply` excludes a forced `prefix`."""
