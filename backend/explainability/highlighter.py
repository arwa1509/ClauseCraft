"""
Entity Highlighter
===================
Produces annotated text with highlighted entities for display in the UI.
Supports HTML and plain-text output with color-coded entity types.
"""

from typing import Optional
from loguru import logger


# Color map for entity types (used in HTML output)
ENTITY_COLORS = {
    "STATUTE": {"bg": "#dbeafe", "text": "#1e40af", "border": "#93c5fd"},
    "PROVISION": {"bg": "#ede9fe", "text": "#5b21b6", "border": "#c4b5fd"},
    "COURT": {"bg": "#fce7f3", "text": "#9d174d", "border": "#f9a8d4"},
    "JUDGE": {"bg": "#fef3c7", "text": "#92400e", "border": "#fcd34d"},
    "CASE_CITATION": {"bg": "#d1fae5", "text": "#065f46", "border": "#6ee7b7"},
    "LEGAL_ACTION": {"bg": "#ffedd5", "text": "#9a3412", "border": "#fdba74"},
    "DATE": {"bg": "#e0e7ff", "text": "#3730a3", "border": "#a5b4fc"},
    "PARTY": {"bg": "#f3e8ff", "text": "#7c3aed", "border": "#c084fc"},
    "ACT": {"bg": "#cffafe", "text": "#155e75", "border": "#67e8f9"},
    "PENALTY": {"bg": "#fee2e2", "text": "#991b1b", "border": "#fca5a5"},
    "JURISDICTION": {"bg": "#ecfccb", "text": "#3f6212", "border": "#bef264"},
    "ORG": {"bg": "#f1f5f9", "text": "#475569", "border": "#94a3b8"},
}


class EntityHighlighter:
    """
    Produces highlighted/annotated text with entity markers.
    Supports multiple output formats for frontend rendering.
    """

    def highlight(
        self,
        text: str,
        entities: list[dict],
        format: str = "annotations",
    ) -> dict | str:
        """
        Highlight entities in text.

        Args:
            text: Input text
            entities: List of entity dicts with text, label, start, end
            format: Output format - "annotations", "html", or "markdown"

        Returns:
            Highlighted text or annotation data
        """
        if not text or not entities:
            return {"text": text, "annotations": []} if format == "annotations" else text

        if format == "html":
            return self._highlight_html(text, entities)
        elif format == "markdown":
            return self._highlight_markdown(text, entities)
        else:
            return self._create_annotations(text, entities)

    def _create_annotations(
        self, text: str, entities: list[dict]
    ) -> dict:
        """
        Create structured annotations for frontend rendering.
        Returns the original text plus annotation markers.
        """
        # Sort entities by start position
        sorted_entities = sorted(entities, key=lambda e: e.get("start", 0))

        # Remove overlapping entities (keep first)
        non_overlapping = []
        last_end = -1
        for entity in sorted_entities:
            start = entity.get("start", -1)
            end = entity.get("end", -1)

            if start < 0:
                # No position info — find by text
                idx = text.lower().find(entity.get("text", "").lower())
                if idx >= 0:
                    start = idx
                    end = idx + len(entity.get("text", ""))
                else:
                    continue

            if start >= last_end:
                non_overlapping.append({
                    "text": entity.get("text", text[start:end]),
                    "label": entity.get("label", "UNKNOWN"),
                    "start": start,
                    "end": end,
                    "color": ENTITY_COLORS.get(
                        entity.get("label", ""), ENTITY_COLORS["ORG"]
                    ),
                })
                last_end = end

        return {
            "text": text,
            "annotations": non_overlapping,
        }

    def _highlight_html(self, text: str, entities: list[dict]) -> str:
        """Produce HTML with color-coded entity spans."""
        annotations = self._create_annotations(text, entities)
        parts = []
        last_idx = 0

        for ann in annotations["annotations"]:
            start = ann["start"]
            end = ann["end"]
            color = ann["color"]
            label = ann["label"]

            # Add text before this entity
            if start > last_idx:
                parts.append(text[last_idx:start])

            # Entity span
            parts.append(
                f'<span class="entity" '
                f'style="background:{color["bg"]};color:{color["text"]};'
                f'border:1px solid {color["border"]};'
                f'padding:2px 6px;border-radius:4px;font-weight:500;" '
                f'data-label="{label}" '
                f'title="{label}">'
                f'{text[start:end]}'
                f'<sup style="font-size:0.7em;margin-left:2px;opacity:0.7;">'
                f'{label}</sup>'
                f'</span>'
            )
            last_idx = end

        # Add remaining text
        if last_idx < len(text):
            parts.append(text[last_idx:])

        return "".join(parts)

    def _highlight_markdown(self, text: str, entities: list[dict]) -> str:
        """Produce markdown with entity labels."""
        annotations = self._create_annotations(text, entities)
        parts = []
        last_idx = 0

        for ann in annotations["annotations"]:
            start = ann["start"]
            end = ann["end"]
            label = ann["label"]

            if start > last_idx:
                parts.append(text[last_idx:start])

            parts.append(f"**{text[start:end]}** `[{label}]`")
            last_idx = end

        if last_idx < len(text):
            parts.append(text[last_idx:])

        return "".join(parts)

    @staticmethod
    def get_entity_legend() -> list[dict]:
        """Get the entity type color legend for the UI."""
        legend = []
        for label, colors in ENTITY_COLORS.items():
            legend.append({
                "label": label,
                "bg": colors["bg"],
                "text": colors["text"],
                "border": colors["border"],
                "description": _ENTITY_DESCRIPTIONS.get(label, label),
            })
        return legend


_ENTITY_DESCRIPTIONS = {
    "STATUTE": "Legal statute or section reference",
    "PROVISION": "Constitutional article or provision",
    "COURT": "Court or tribunal name",
    "JUDGE": "Judge or justice name",
    "CASE_CITATION": "Case citation reference",
    "LEGAL_ACTION": "Legal action or proceeding type",
    "DATE": "Date reference",
    "PARTY": "Party to the case",
    "ACT": "Name of a legal act or code",
    "PENALTY": "Penalty, fine, or sentence",
    "JURISDICTION": "Jurisdiction or geographical area",
    "ORG": "Organization",
}
