from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


CONTROLLER_MODES = frozenset({"frozen_baseline", "minimal_tool", "mger"})
EVIDENCE_VISIBILITIES = frozenset({"none", "candidates_only", "full"})
MEASUREMENT_CONTROLS = frozenset({"none", "blind_prior", "caption_only"})


@dataclass(frozen=True)
class Phase5Protocol:
    controller_mode: str = "mger"
    controller_evidence_visibility: str = "full"
    measurement_control: str = "none"

    def __post_init__(self) -> None:
        controller_mode = str(self.controller_mode or "mger").strip().casefold()
        evidence_visibility = str(
            self.controller_evidence_visibility or "full"
        ).strip().casefold()
        measurement_control = str(
            self.measurement_control or "none"
        ).strip().casefold()
        if controller_mode not in CONTROLLER_MODES:
            raise ValueError(f"unsupported controller_mode: {controller_mode}")
        if evidence_visibility not in EVIDENCE_VISIBILITIES:
            raise ValueError(
                "unsupported controller_evidence_visibility: "
                f"{evidence_visibility}"
            )
        if measurement_control not in MEASUREMENT_CONTROLS:
            raise ValueError(
                f"unsupported measurement_control: {measurement_control}"
            )
        if measurement_control != "none" and controller_mode != "frozen_baseline":
            raise ValueError(
                "measurement controls must use controller_mode=frozen_baseline"
            )
        if measurement_control == "blind_prior" and evidence_visibility != "none":
            raise ValueError(
                "blind_prior requires controller_evidence_visibility=none"
            )
        object.__setattr__(self, "controller_mode", controller_mode)
        object.__setattr__(
            self,
            "controller_evidence_visibility",
            evidence_visibility,
        )
        object.__setattr__(self, "measurement_control", measurement_control)

    @property
    def arm(self) -> str:
        if self.measurement_control != "none":
            return self.measurement_control
        if self.controller_mode == "frozen_baseline":
            return "frozen_baseline"
        if self.controller_mode == "minimal_tool":
            return (
                "minimal_candidates"
                if self.controller_evidence_visibility == "candidates_only"
                else "minimal_passive"
            )
        return "mger"

    @property
    def allowed_inspection_modes(self) -> frozenset[str] | None:
        if self.measurement_control == "blind_prior":
            return frozenset()
        if self.measurement_control == "caption_only":
            return frozenset({"search_caption"})
        return None

    def to_dict(self) -> dict[str, str]:
        return {
            "phase5_arm": self.arm,
            "controller_mode": self.controller_mode,
            "controller_evidence_visibility": self.controller_evidence_visibility,
            "measurement_control": self.measurement_control,
        }


def blind_prior_prompt(question: str) -> str:
    return (
        "Answer the question concisely using only your prior knowledge. "
        "You cannot inspect the video, captions, audio, frames, metadata, or external tools. "
        "Return only the answer, with no explanation.\n"
        f"Question: {str(question)}"
    )


def inspection_mode_policy_errors(
    tasks: Sequence[Any],
    *,
    allowed_modes: frozenset[str] | None,
) -> tuple[dict[str, Any], ...]:
    if allowed_modes is None:
        return ()
    errors: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        mode = _task_value(task, "inspection_mode", "window").strip().casefold()
        if mode in allowed_modes:
            continue
        errors.append(
            {
                "code": "inspection_mode_forbidden",
                "task_index": index,
                "inspection_mode": mode,
                "allowed_inspection_modes": sorted(allowed_modes),
            }
        )
    return tuple(errors)


def _task_value(task: Any, key: str, default: str = "") -> str:
    if isinstance(task, Mapping):
        return str(task.get(key, default) or default)
    return str(getattr(task, key, default) or default)
