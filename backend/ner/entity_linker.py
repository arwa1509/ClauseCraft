"""
Entity Linker
==============
Links extracted entities to specific documents, sections, and related entities.
Supports cross-reference resolution across document corpus.
"""

from typing import Optional
from loguru import logger


class EntityLinker:
    """
    Links named entities to their source documents and sections.
    Resolves entity mentions across chunks and documents.
    """

    def __init__(self):
        # Entity knowledge base: canonical_name → info
        self.knowledge_base = {}

    def link_entities(
        self,
        entities: list[dict],
        chunk_metadata: dict,
    ) -> list[dict]:
        """
        Link entities to their source locations and related entities.

        Args:
            entities: List of entity dicts from NER
            chunk_metadata: Metadata about the chunk (doc_name, section, etc.)

        Returns:
            Entities with added linking information
        """
        linked = []

        for entity in entities:
            linked_entity = {
                **entity,
                "linked_to": {
                    "document": chunk_metadata.get("doc_name", "unknown"),
                    "section": chunk_metadata.get("section"),
                    "section_num": chunk_metadata.get("section_num"),
                    "page_num": chunk_metadata.get("page_num"),
                },
                "canonical_name": self._canonicalize(entity["text"], entity["label"]),
                "related_entities": [],
            }

            # Register in knowledge base
            canonical = linked_entity["canonical_name"]
            if canonical not in self.knowledge_base:
                self.knowledge_base[canonical] = {
                    "label": entity["label"],
                    "mentions": [],
                    "documents": set(),
                }

            self.knowledge_base[canonical]["mentions"].append(entity["text"])
            self.knowledge_base[canonical]["documents"].add(
                chunk_metadata.get("doc_name", "unknown")
            )

            linked.append(linked_entity)

        return linked

    def _canonicalize(self, text: str, label: str) -> str:
        """
        Create a canonical form for an entity to enable linking.
        E.g., "Sec. 302", "Section 302", "S. 302" → "SECTION_302"
        """
        import re

        text_clean = text.strip()

        if label == "STATUTE":
            # Normalize section references
            match = re.search(r"(\d+[A-Za-z]?)", text_clean)
            if match:
                num = match.group(1).upper()
                # Extract act name if present
                act_match = re.search(
                    r"(?:of|under)\s+(?:the\s+)?(.+)", text_clean, re.IGNORECASE
                )
                act = act_match.group(1).strip() if act_match else ""
                return f"SECTION_{num}_{act}".strip("_")
            return text_clean.upper().replace(" ", "_")

        elif label == "PROVISION":
            match = re.search(r"(\d+[A-Za-z]?)", text_clean)
            if match:
                return f"ARTICLE_{match.group(1).upper()}"
            return text_clean.upper().replace(" ", "_")

        elif label == "COURT":
            # Normalize court names
            text_lower = text_clean.lower()
            if "supreme" in text_lower:
                return "SUPREME_COURT_OF_INDIA"
            if "high court" in text_lower:
                # Extract jurisdiction
                match = re.search(r"high\s+court\s+of\s+(\w+)", text_lower)
                if match:
                    return f"HIGH_COURT_{match.group(1).upper()}"
                return "HIGH_COURT"
            return text_clean.upper().replace(" ", "_")

        elif label == "JUDGE":
            # Normalize: "Justice R.M. Lodha" → "JUSTICE_RM_LODHA"
            name = re.sub(r"(?:Hon'?ble|Honourable|Chief)?\s*Justice\s*", "", text_clean)
            name = re.sub(r"\.", "", name).strip()
            return f"JUSTICE_{name.upper().replace(' ', '_')}"

        else:
            return text_clean.upper().replace(" ", "_")

    def find_related(self, entity_canonical: str) -> list[str]:
        """Find entities related to the given one based on co-occurrence."""
        if entity_canonical not in self.knowledge_base:
            return []

        # Find entities appearing in the same documents
        target_docs = self.knowledge_base[entity_canonical]["documents"]
        related = []

        for name, info in self.knowledge_base.items():
            if name == entity_canonical:
                continue
            if info["documents"] & target_docs:
                related.append(name)

        return related[:10]  # Limit to top 10

    def get_entity_info(self, canonical_name: str) -> Optional[dict]:
        """Get information about an entity from the knowledge base."""
        info = self.knowledge_base.get(canonical_name)
        if info:
            return {
                "canonical_name": canonical_name,
                "label": info["label"],
                "mention_count": len(info["mentions"]),
                "documents": list(info["documents"]),
                "mentions": list(set(info["mentions"]))[:5],
            }
        return None
