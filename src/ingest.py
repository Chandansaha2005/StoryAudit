"""ingest.py
Document loading and simple Pathway-compatible helpers.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

# Try to import Pathway but stay compatible if unavailable
try:
    import pathway as pw  # type: ignore
    HAS_PATHWAY = True
except Exception:
    pw = None
    HAS_PATHWAY = False

logger = logging.getLogger(__name__)


class DocumentIngestion:
    """Load files and create Pathway tables."""

    def __init__(self) -> None:
        self.documents: Dict[str, str] = {}

    def load_document(self, file_path: Path) -> str:
        """Load text from file."""
        logger.info(f"Loading document: {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if not content.strip():
            raise ValueError(f"Document is empty: {file_path}")

        # Cache and return
        self.documents[file_path.name] = content
        return content

    def create_pathway_table(self, documents: Dict[str, str]):
        """Create Pathway table fallback."""
        logger.info(f"Creating Pathway table for {len(documents)} documents")
        if not HAS_PATHWAY:
            # Return a simple dict fallback
            return {"id": list(documents.keys()), "text": list(documents.values())}

        # If Pathway is available, create a table object
        rows = [{"id": k, "text": v} for k, v in documents.items()]
        # Minimal pathway usage: build a table from rows
        table = pw.Table.from_rows(rows) if hasattr(pw, "Table") else rows
        return table

    def validate_document(self, content: str, min_words: int = 100) -> bool:
        """Check minimum word count."""
        word_count = len(content.split())
        if word_count < min_words:
            logger.warning(f"Document too short: {word_count} words (min: {min_words})")
            return False
        return True


class NarrativeLoader:
    """Load narratives from directory."""

    def __init__(self, narratives_dir: Path) -> None:
        self.narratives_dir = Path(narratives_dir)
        self.ingestion = DocumentIngestion()

    def load_narrative(self, story_id: str) -> str:
        """Load narrative by story ID."""
        possible_names = [f"story_{story_id}.txt", f"{story_id}.txt", f"narrative_{story_id}.txt"]

        for name in possible_names:
            file_path = self.narratives_dir / name
            if file_path.exists():
                return self.ingestion.load_document(file_path)

        raise FileNotFoundError(f"Narrative not found for story id: {story_id}")

    def get_narrative_metadata(self, content: str) -> Dict[str, int]:
        """Get narrative statistics."""
        words = content.split()
        return {"word_count": len(words), "char_count": len(content), "line_count": content.count("\n")}


class BackstoryLoader:
    """Load backstories from directory."""

    def __init__(self, backstories_dir: Path) -> None:
        self.backstories_dir = Path(backstories_dir)
        self.ingestion = DocumentIngestion()

    def load_backstory(self, story_id: str) -> str:
        """Load backstory by story ID."""
        possible_names = [f"backstory_{story_id}.txt", f"{story_id}_backstory.txt", f"{story_id}.txt"]

        for name in possible_names:
            file_path = self.backstories_dir / name
            if file_path.exists():
                return self.ingestion.load_document(file_path)

        raise FileNotFoundError(f"Backstory not found for story id: {story_id}")


class PathwayDocumentStore:
    """Container for documents and tables."""

    def __init__(self) -> None:
        self.documents: Dict[str, str] = {}
        self.pathway_table = None

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> None:
        self.documents[doc_id] = text

    def build_pathway_table(self):
        ingestion = DocumentIngestion()
        self.pathway_table = ingestion.create_pathway_table(self.documents)
        return self.pathway_table
