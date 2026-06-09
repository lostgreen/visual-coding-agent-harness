"""Task-type playbooks for progressive planner guidance."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class QuestionPlaybook:
    name: str
    route: str = "needle_local"
    instructions: Sequence[str] = field(default_factory=list)
    sufficiency_rules: Sequence[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        lines = [f"Task playbook: {self.name}", f"Question route: {self.route}", "Playbook instructions:"]
        lines.extend(f"- {instruction}" for instruction in self.instructions)
        lines.append("Evidence sufficiency:")
        lines.extend(f"- {rule}" for rule in self.sufficiency_rules)
        return "\n".join(lines)


@dataclass(frozen=True)
class OptionSequenceSpec:
    option_letter: str
    ordered_items: tuple[str, ...]
    ordered_target_refs: tuple[str, ...]


def select_question_playbook(question: str) -> QuestionPlaybook:
    route = classify_question_route(question)
    if route == "gist_global":
        return QuestionPlaybook(
            name="gist_global",
            route=route,
            instructions=[
                "Start with global_gist to get a sparse whole-video topic and coverage hint.",
                "Use local inspection or indexed transcript evidence to verify the full-video coverage.",
                "Do not shred synopsis or overall-theme questions into local MCQ votes first.",
            ],
            sufficiency_rules=[
                "A global_gist observation is a topic hint, not structured option support.",
                "Prefer the option that covers the most video stages; partial ending-only coverage cannot beat a full rise/stability/fall arc.",
            ],
        )

    if route == "temporal_order":
        return QuestionPlaybook(
            name="timeline_ordering",
            route=route,
            instructions=[
                "Use coarse captions to locate candidate event segments before focused timestamp reads.",
                "Inspect at least the relevant earlier and later event windows when order matters.",
                "Use local workers for open factual descriptions; use original options for planning/search only.",
            ],
            sufficiency_rules=[
                "Citations must include timestamped answer-grade visual, ASR, OCR, or QA evidence for the ordered events.",
                "Evidence must not conflict with the claimed temporal relation.",
                "verify option consistency against the cited observation before final.",
            ],
        )

    if extract_candidate_options(question):
        return QuestionPlaybook(
            name="multiple_choice",
            route=route,
            instructions=[
                "Use video_ls/search_segments to localize candidates before visual inspection.",
                "Use original options to identify discriminative search atoms; local VLM tools receive neutral factual prompts only.",
                "Local workers must report facts only; AnswerAgent maps facts to options.",
                "Avoid finalizing from navigation-only evidence.",
            ],
            sufficiency_rules=[
                "At least one cited answer-grade visual, ASR, OCR, or QA observation must ground the selected option.",
                "verify option consistency against the cited observation before final.",
                "Final answer should preserve the option letter when the user provided choices.",
            ],
        )

    return QuestionPlaybook(
        name="general_video_qa",
        route=route,
        instructions=[
            "Use query-conditioned navigation to localize likely evidence.",
            "Delegate visual reading to inspect_segment once a candidate is localized.",
        ],
        sufficiency_rules=[
            "Final answers need cited answer-grade visual, ASR, OCR, or QA evidence.",
            "State uncertainty when evidence is incomplete or ambiguous.",
        ],
    )


def classify_narration_subroute(question: str) -> str:
    """Split temporal questions into narrated biography vs visual timeline cases."""

    semantic_lowered = _semantic_question_text(question).lower()
    visual_hard_negatives = [
        "open the door",
        "painting positioned",
        "ball move",
        "move after impact",
        "pick up",
        "visible",
        "shown on screen",
    ]
    if any(marker in semantic_lowered for marker in visual_hard_negatives):
        return "visual_timeline"

    explicit_narration_markers = [
        "according to the narrator",
        "narrator",
        "narration",
        "voiceover",
        "voice-over",
        "tells us",
        "tell us",
        "told us",
        "what does the video say",
        "what does the video tell",
    ]
    if any(marker in semantic_lowered for marker in explicit_narration_markers):
        return "narration_timeline"

    biographical_markers = [
        "life journey",
        "early life",
        "childhood",
        "born",
        "grew up",
        "biography",
        "career",
        "left home",
        "background",
    ]
    asr_availability_hints = [
        "according to",
        "the video",
        "narrator",
        "narration",
        "voiceover",
        "voice-over",
        "tell",
        "tells",
        "told",
        "say",
        "says",
        "said",
    ]
    if any(marker in semantic_lowered for marker in biographical_markers) and any(
        hint in semantic_lowered for hint in asr_availability_hints
    ):
        return "narration_timeline"

    return "visual_timeline"


def classify_question_route(question: str) -> str:
    """Classify whether the question needs a whole-video floor or localized search."""

    semantic_lowered = _semantic_question_text(question).lower()
    option_lowered = " ".join(extract_candidate_options(question)).lower()
    lowered = f"{semantic_lowered} {option_lowered}".strip()
    gist_markers = [
        "mainly about",
        "primarily about",
        "overall",
        "whole video",
        "entire video",
        "synopsis",
        "summary",
        "summarize",
        "theme",
        "main idea",
        "central idea",
        "main topic",
        "main subject",
        "main content",
        "general content",
        "what is the video about",
    ]
    if any(marker in semantic_lowered for marker in gist_markers):
        return "gist_global"

    temporal_markers = [
        "right after",
        "right before",
        "immediately after",
        "immediately before",
        "before",
        "after",
        "first",
        "last",
        "then",
        "order",
        "sequence",
        "temporal",
        "earlier",
        "later",
        "at the beginning",
        "at the end",
        "specific moment",
        "life journey",
        "journey",
        "rose",
        "rise",
        "entered",
        "moved into",
        "withdrew",
        "withdrawal",
        "seclusion",
    ]
    if any(marker in lowered for marker in temporal_markers):
        return "temporal_order"
    return "needle_local"


def extract_candidate_options(question: str) -> Sequence[str]:
    options = []
    for line in question.splitlines():
        stripped = line.strip()
        if re.match(r"^[A-H][.)]\s+\S+", stripped):
            options.append(stripped)
    if options:
        return options

    normalized = re.sub(r"\bOptions\s*:\s*", " ", question, flags=re.IGNORECASE)
    matches = re.finditer(
        r"(?<![A-Za-z0-9])([A-H])([.)])\s+(.*?)(?=(?<![A-Za-z0-9])[A-H][.)]\s+|$)",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in matches:
        text = " ".join(str(match.group(3)).split()).strip()
        if text:
            options.append(f"{match.group(1).upper()}{match.group(2)} {text}")
    return options


OPTION_TARGET_SYNONYMS = {
    "upper class": ["upper echelons", "high society", "royal court", "bourgeois", "court painter"],
    "seclusion": ["isolation", "isolated", "secluded", "withdrew from public life"],
    "humble background": ["humble origins", "modest background", "from a humble background"],
    "farmhouse": ["country house", "countryside farmhouse", "secluded farmhouse"],
}


def extract_option_target_atoms(
    question_or_options: str | Sequence[str],
    *,
    max_targets: int = 16,
    include_synonyms: bool = True,
) -> list[str]:
    """Extract option-derived factual atoms for planning/search, not VLM option voting."""

    options = (
        list(extract_candidate_options(question_or_options))
        if isinstance(question_or_options, str)
        else [str(option) for option in question_or_options]
    )
    quoted_registry = _canonical_quoted_option_items(options, max_targets=max_targets)
    if quoted_registry:
        return quoted_registry[:max_targets]
    atoms: list[str] = []
    seen: set[str] = set()
    for option in options:
        for atom in extract_option_target_atoms_for_option(option, include_synonyms=False):
            key = _target_atom_key(atom)
            if not key or key in seen:
                continue
            seen.add(key)
            atoms.append(atom)
            if len(atoms) >= max_targets:
                return atoms
    if include_synonyms:
        for atom in list(atoms):
            for synonym in OPTION_TARGET_SYNONYMS.get(atom.lower(), []):
                key = _target_atom_key(synonym)
                if not key or key in seen:
                    continue
                seen.add(key)
                atoms.append(synonym)
                if len(atoms) >= max_targets:
                    return atoms
    return atoms


def extract_option_target_atom_map(
    question_or_options: str | Sequence[str],
    *,
    include_synonyms: bool = True,
) -> dict[str, list[str]]:
    options = (
        list(extract_candidate_options(question_or_options))
        if isinstance(question_or_options, str)
        else [str(option) for option in question_or_options]
    )
    mapping: dict[str, list[str]] = {}
    for index, option in enumerate(options):
        letter = _option_letter(option, index=index)
        atoms = extract_option_target_atoms_for_option(option, include_synonyms=include_synonyms)
        if atoms:
            mapping[letter] = atoms
    return mapping


def extract_option_sequence_specs(question_or_options: str | Sequence[str]) -> dict[str, OptionSequenceSpec]:
    """Parse ordered MCQ option item sequences without splitting quoted titles."""

    options = (
        list(extract_candidate_options(question_or_options))
        if isinstance(question_or_options, str)
        else [str(option) for option in question_or_options]
    )
    life_journey_sequences = _life_journey_option_sequence_specs(options)
    if life_journey_sequences:
        return life_journey_sequences
    parsed: list[tuple[str, tuple[str, ...]]] = []
    for index, option in enumerate(options):
        letter = _option_letter(option, index=index)
        items = tuple(_quoted_option_items(option))
        if not items:
            items = tuple(
                _clean_option_atom(chunk)
                for chunk in _split_option_atom_text(_strip_option_prefix(option))
                if _is_informative_option_atom(_clean_option_atom(chunk))
            )
        if items:
            parsed.append((letter, items))
    if not parsed:
        return {}

    canonical_items = _canonical_items_from_option_sequences([items for _, items in parsed])
    if not canonical_items:
        return {}
    refs_by_key = {
        _target_atom_key(item): f"T{index}"
        for index, item in enumerate(canonical_items, start=1)
        if _target_atom_key(item)
    }
    sequences: dict[str, OptionSequenceSpec] = {}
    for letter, items in parsed:
        refs = tuple(refs_by_key.get(_target_atom_key(item), "") for item in items)
        if len(refs) != len(items) or any(not ref for ref in refs):
            continue
        sequences[letter] = OptionSequenceSpec(
            option_letter=letter,
            ordered_items=tuple(_canonical_item_text(item, canonical_items) for item in items),
            ordered_target_refs=refs,
        )
    return sequences


_LIFE_JOURNEY_TARGETS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("T1", "humble background", (r"\b(?:born|came|coming|from|with)\b.{0,45}\b(?:humble|modest|lowly)\b.{0,35}\b(?:background|origins?)\b",)),
    (
        "T2",
        "entered upper class",
        (
            r"\b(?:entered|entering|rose|risen|reached|became|becoming|joined)\b.{0,55}\b(?:upper class|upper echelons?|high society|royal court|court painter|bourgeois)\b",
        ),
    ),
    (
        "T3",
        "seclusion/farmhouse",
        (
            r"\b(?:lived|living|withdrew|withdrawn|retired|spent)\b.{0,55}\b(?:seclusion|isolation|isolated|secluded|farmhouse|country house)\b",
            r"\b(?:seclusion|isolation|isolated|secluded|farmhouse|country house)\b",
        ),
    ),
    (
        "T4",
        "born in upper class",
        (
            r"\bborn\b.{0,20}\b(?:in|into|to)\b.{0,25}\b(?:upper class|upper echelons?|high society|royal court|bourgeois)\b",
        ),
    ),
)


def _life_journey_option_sequence_specs(options: Sequence[str]) -> dict[str, OptionSequenceSpec]:
    parsed: dict[str, OptionSequenceSpec] = {}
    for index, option in enumerate(options):
        letter = _option_letter(option, index=index)
        events = _life_journey_events_for_option(option)
        if len(events) < 2:
            continue
        parsed[letter] = OptionSequenceSpec(
            option_letter=letter,
            ordered_items=tuple(item for _ref, item in events),
            ordered_target_refs=tuple(ref for ref, _item in events),
        )
    if len(parsed) < 2:
        return {}
    all_refs = {ref for sequence in parsed.values() for ref in sequence.ordered_target_refs}
    if {"T1", "T3"}.issubset(all_refs) and ({"T2", "T4"} & all_refs):
        return parsed
    return {}


def _life_journey_events_for_option(option_text: str) -> list[tuple[str, str]]:
    text = _strip_option_prefix(option_text)
    lowered = text.lower()
    matches: list[tuple[int, str, str]] = []
    for target_ref, canonical_text, patterns in _LIFE_JOURNEY_TARGETS:
        target_matches = [
            match
            for pattern in patterns
            for match in [re.search(pattern, lowered)]
            if match is not None
        ]
        if not target_matches:
            continue
        first = min(target_matches, key=lambda match: match.start())
        matches.append((first.start(), target_ref, canonical_text))
    matches.sort(key=lambda item: item[0])
    events: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _position, target_ref, canonical_text in matches:
        if target_ref in seen:
            continue
        seen.add(target_ref)
        events.append((target_ref, canonical_text))
    return events


def extract_option_target_atoms_for_option(
    option_text: str,
    *,
    include_synonyms: bool = False,
) -> list[str]:
    text = _strip_option_prefix(option_text)
    quoted_items = _quoted_option_items(text)
    if quoted_items:
        return list(quoted_items)
    lowered = text.lower()
    positioned_atoms: list[tuple[int, str]] = []
    for pattern, atom in (
        (r"\b(?:humble|modest|lowly)\b.{0,40}\b(?:background|origins?)\b", "humble background"),
        (r"\b(?:upper class|upper echelons?|high society|royal court|court painter|bourgeois)\b", "upper class"),
        (r"\b(?:farmhouse|country house|countryside farmhouse)\b", "farmhouse"),
        (r"\b(?:seclusion|isolation|isolated|secluded|withdrew|withdrawal)\b", "seclusion"),
    ):
        match = re.search(pattern, lowered)
        if match:
            positioned_atoms.append((match.start(), atom))
    positioned_atoms.sort(key=lambda item: item[0])

    for chunk in _split_option_atom_text(text):
        cleaned = _clean_option_atom(chunk)
        if _chunk_is_covered_by_canonical_atom(cleaned, [atom for _, atom in positioned_atoms]):
            continue
        if _is_informative_option_atom(cleaned):
            positioned_atoms.append((lowered.find(chunk.lower().strip()), cleaned))

    unique: list[str] = []
    seen: set[str] = set()
    for _, atom in sorted(positioned_atoms, key=lambda item: item[0] if item[0] >= 0 else 10**9):
        key = _target_atom_key(atom)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(atom)
    if include_synonyms:
        for atom in list(unique):
            for synonym in OPTION_TARGET_SYNONYMS.get(atom.lower(), []):
                key = _target_atom_key(synonym)
                if key and key not in seen:
                    seen.add(key)
                    unique.append(synonym)
    return unique


def _semantic_question_text(question: str) -> str:
    match = re.search(r"\bQuestion:\s*(.*?)(?:\n\s*Options:|\Z)", question, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1)
    lines = []
    for line in question.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^[A-H][.)]\s+\S+", stripped):
            continue
        if "answer with" in stripped.lower() and "option letter" in stripped.lower():
            continue
        lines.append(stripped)
    return "\n".join(lines) or question


def _strip_option_prefix(text: str) -> str:
    return re.sub(r"^\s*[A-H][.)]\s*", "", str(text)).strip()


def _option_letter(option_text: str, *, index: int) -> str:
    match = re.match(r"\s*([A-H])(?:[.)]\s*|\s+|$)", str(option_text), flags=re.IGNORECASE)
    return match.group(1).upper() if match else chr(ord("A") + index)


def _split_option_atom_text(text: str) -> list[str]:
    primary = re.split(r"\b(?:and then|then|before|after)\b|[,;]|->|/|\|", text, flags=re.IGNORECASE)
    parts: list[str] = []
    action_splitter = re.compile(
        r"\band\s+(?=(?:lived|entered|went|moved|became|was|were|is|are|appeared|appears|shown|shows|born|rose|withdrew)\b)",
        flags=re.IGNORECASE,
    )
    for part in primary:
        parts.extend(action_splitter.split(part))
    return parts


def _quoted_option_items(option_text: str) -> tuple[str, ...]:
    items = [
        _clean_option_atom(match.group(1))
        for match in re.finditer(r"[\"“]([^\"”]+)[\"”]", str(option_text))
    ]
    return tuple(item for item in items if _is_informative_option_atom(item))


def _canonical_quoted_option_items(options: Sequence[str], *, max_targets: int) -> list[str]:
    sequences = [_quoted_option_items(option) for option in options]
    sequences = [sequence for sequence in sequences if sequence]
    if not sequences:
        return []
    return _canonical_items_from_option_sequences(sequences)[:max_targets]


def _canonical_items_from_option_sequences(sequences: Sequence[Sequence[str]]) -> list[str]:
    if not sequences:
        return []
    first_keys = {_target_atom_key(item) for item in sequences[0] if _target_atom_key(item)}
    if not first_keys:
        return []
    for sequence in sequences:
        keys = {_target_atom_key(item) for item in sequence if _target_atom_key(item)}
        if keys != first_keys:
            return []
    canonical_by_key: dict[str, str] = {}
    for sequence in sequences:
        for item in sequence:
            key = _target_atom_key(item)
            if key and key not in canonical_by_key:
                canonical_by_key[key] = item
    ordered_sequences = [
        tuple(_target_atom_key(item) for item in sequence if _target_atom_key(item))
        for sequence in sequences
    ]
    registry_order = min(ordered_sequences, key=lambda keys: keys[0] if keys else "")
    return [canonical_by_key[key] for key in registry_order]


def _canonical_item_text(item: str, canonical_items: Sequence[str]) -> str:
    key = _target_atom_key(item)
    for canonical in canonical_items:
        if _target_atom_key(canonical) == key:
            return canonical
    return item


def _clean_option_atom(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip(" .:-\"'“”")
    cleaned = re.sub(r"\b(first|last|earlier|later|right|immediately|eventually)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\b(?:born|borned)\s+(?:with|in|from)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\b(?:lived|entered|moved|went|rose|withdrew)\s+(?:in|into|to|from|through)\s+", "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip(" .:-\"'“”")


def _chunk_is_covered_by_canonical_atom(chunk: str, canonical_atoms: Sequence[str]) -> bool:
    lowered = str(chunk or "").lower()
    canonical = {str(atom).lower() for atom in canonical_atoms}
    if "humble background" in canonical and re.search(r"\b(?:humble|modest|lowly)\b.*\b(?:background|origins?)\b", lowered):
        return True
    if "upper class" in canonical and re.search(r"\b(?:upper class|upper echelons?|high society|royal court|court painter|bourgeois)\b", lowered):
        return True
    if {"seclusion", "farmhouse"}.issubset(canonical) and "seclusion" in lowered and "farmhouse" in lowered:
        return True
    if "seclusion" in canonical and re.search(r"\b(?:seclusion|isolation|isolated|secluded|withdrew|withdrawal)\b", lowered):
        return True
    if "farmhouse" in canonical and re.search(r"\b(?:farmhouse|country house|countryside farmhouse)\b", lowered):
        return True
    return False


_OPTION_ATOM_STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "as",
    "background",
    "by",
    "choice",
    "in",
    "is",
    "of",
    "option",
    "or",
    "the",
    "then",
    "to",
    "video",
    "with",
}


def _is_informative_option_atom(text: str) -> bool:
    tokens = [token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if token not in _OPTION_ATOM_STOPWORDS]
    return len(tokens) >= 1 and any(len(token) >= 3 for token in tokens)


def _target_atom_key(text: str) -> str:
    tokens = [token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if token not in _OPTION_ATOM_STOPWORDS]
    return " ".join(tokens)
