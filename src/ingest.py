"""
ingest.py
Pathway-based document ingestion and management
"""

import logging
from pathlib import Path
from typing import Optional
import pathway as pw

logger = logging.getLogger(__name__)


class DocumentIngestion:
    """Handles document loading and management using Pathway."""
    
    def __init__(self):
        """Initialize document ingestion system."""
        self.documents = {}
        
    def load_document(self, file_path: Path) -> str:
        """
        Load a single document using Pathway's file system connector.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Document text content
        """
        logger.info(f"Loading document: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                raise ValueError(f"Document is empty: {file_path}")
            
            logger.info(f"Loaded {len(content):,} characters from {file_path.name}")
            return content
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise
    
    def create_pathway_table(self, documents: dict[str, str]):
        """
        Create a Pathway table from documents for stream processing.
        
        Args:
            documents: Dict mapping document_id -> content
            
        Returns:
            Pathway Table with document data
        """
        logger.info(f"Creating Pathway table with {len(documents)} documents")
        
        # Convert documents to Pathway-compatible format
        data = [
            {"id": doc_id, "text": content}
            for doc_id, content in documents.items()
        ]
        
        # Create Pathway table from data
        # In production, this would use pathway.io connectors
        # For hackathon, we use in-memory table creation
        table = pw.debug.table_from_rows(
            schema=pw.schema_builder({
                "id": pw.column_definition(dtype=str),
                "text": pw.column_definition(dtype=str)
            }),
            rows=[(d["id"], d["text"]) for d in data]
        )
        
        logger.info("Pathway table created successfully")
        return table
    
    def validate_document(self, content: str, min_words: int = 1000) -> bool:
        """
        Validate document meets minimum requirements.
        
        Args:
            content: Document text
            min_words: Minimum word count required
            
        Returns:
            True if valid, False otherwise
        """
        word_count = len(content.split())
        
        if word_count < min_words:
            logger.warning(f"Document too short: {word_count} words (min: {min_words})")
            return False
        
        return True


class NarrativeLoader:
    """Specialized loader for narrative documents (novels)."""
    
    def __init__(self, narratives_dir: Path):
        """
        Initialize narrative loader.
        
        Args:
            narratives_dir: Directory containing narrative files
        """
        self.narratives_dir = narratives_dir
        self.ingestion = DocumentIngestion()
        
    def load_narrative(self, story_id: str) -> str:
        """
        Load narrative by story ID.
        
        Args:
            story_id: Story identifier (e.g., "story_1" or "1")
            
        Returns:
            Full narrative text
        """
        # Try multiple filename patterns
        possible_names = [
            f"story_{story_id}.txt",
            f"{story_id}.txt",
            f"narrative_{story_id}.txt"
        ]
        
        for name in possible_names:
            file_path = self.narratives_dir / name
            if file_path.exists():
                narrative = self.ingestion.load_document(file_path)
                
                # Validate it's a proper novel (100k+ words expected)
                if not self.ingestion.validate_document(narrative, min_words=10000):
                    logger.warning(f"Narrative {story_id} is unusually short")
                
                return narrative
        
        raise FileNotFoundError(
            f"Narrative not found for story_id={story_id}. "
            f"Tried: {possible_names} in {self.narratives_dir}"
        )
    
    def get_narrative_metadata(self, content: str) -> dict:
        """
        Extract basic metadata from narrative.
        
        Args:
            content: Narrative text
            
        Returns:
            Dict with word count, character count, etc.
        """
        words = content.split()
        return {
            "word_count": len(words),
            "char_count": len(content),
            "line_count": content.count('\n'),
            "estimated_pages": len(words) // 250  # ~250 words per page
        }


class BackstoryLoader:
    """Specialized loader for backstory documents."""
    
    def __init__(self, backstories_dir: Path):
        """
        Initialize backstory loader.
        
        Args:
            backstories_dir: Directory containing backstory files
        """
        self.backstories_dir = backstories_dir
        self.ingestion = DocumentIngestion()
        
    def load_backstory(self, story_id: str) -> str:
        """
        Load backstory by story ID.
        
        Args:
            story_id: Story identifier matching the narrative
            
        Returns:
            Backstory text
        """
        # Try multiple filename patterns
        possible_names = [
            f"backstory_{story_id}.txt",
            f"{story_id}_backstory.txt",
            f"{story_id}.txt"
        ]
        
        for name in possible_names:
            file_path = self.backstories_dir / name
            if file_path.exists():
                backstory = self.ingestion.load_document(file_path)
                
                # Validate backstory is substantial enough
                if not self.ingestion.validate_document(backstory, min_words=100):
                    logger.warning(f"Backstory {story_id} is very short")
                
                return backstory
        
        raise FileNotFoundError(
            f"Backstory not found for story_id={story_id}. "
            f"Tried: {possible_names} in {self.backstories_dir}"
        )
    
    def preprocess_backstory(self, content: str) -> str:
        """
        Preprocess backstory text for claim extraction.
        
        Args:
            content: Raw backstory text
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        content = '\n'.join(line.strip() for line in content.split('\n'))
        
        # Remove multiple consecutive blank lines
        while '\n\n\n' in content:
            content = content.replace('\n\n\n', '\n\n')
        
        return content.strip()


# ============================================================================
# PATHWAY INTEGRATION UTILITIES
# ============================================================================

class PathwayDocumentStore:
    """
    Wrapper for Pathway-based document storage and querying.
    Provides a simple interface for document management.
    """
    
    def __init__(self):
        """Initialize document store."""
        self.documents = {}
        self.pathway_table: Optional[pw.Table] = None
        
    def add_document(self, doc_id: str, content: str, metadata: dict = None):
        """
        Add a document to the store.
        
        Args:
            doc_id: Unique document identifier
            content: Document text content
            metadata: Optional metadata dict
        """
        self.documents[doc_id] = {
            "content": content,
            "metadata": metadata or {}
        }
        logger.debug(f"Added document {doc_id} ({len(content)} chars)")
    
    def build_pathway_table(self):
        """Build Pathway table from stored documents."""
        if not self.documents:
            raise ValueError("No documents to build table from")
        
        ingestion = DocumentIngestion()
        doc_contents = {doc_id: data["content"] 
                       for doc_id, data in self.documents.items()}
        
        self.pathway_table = ingestion.create_pathway_table(doc_contents)
        logger.info("Pathway document store initialized")
    
    def get_document(self, doc_id: str) -> str:
        """Retrieve document by ID."""
        if doc_id not in self.documents:
            raise KeyError(f"Document {doc_id} not found")
        return self.documents[doc_id]["content"]
    
    def get_all_document_ids(self) -> list[str]:
        """Get list of all document IDs."""
        return list(self.documents.keys())