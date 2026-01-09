"""
cache_manager.py
Efficient caching system for StoryAudit with persistent storage.
Implements Track-A compliance with evidence tracking and reproducibility.
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Metadata for cached items."""
    key: str
    category: str  # 'novel', 'chunks', 'embeddings', 'results'
    created_at: float
    ttl: Optional[float] = None  # Time to live in seconds
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl


class CacheManager:
    """
    Manages caching of expensive operations with persistent storage.
    
    Track-A Compliance:
    - All cache operations are logged for reproducibility
    - Cache hits/misses tracked for evidence chain
    - Timestamped entries for audit trail
    - Organized by category for clear separation of concerns
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, enable_persistent: bool = True):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for persistent cache storage
            enable_persistent: Enable persistent disk caching
        """
        if cache_dir is None:
            cache_dir = Path.cwd() / ".storyaudit_cache"
        
        self.cache_dir = Path(cache_dir)
        self.enable_persistent = enable_persistent
        self.memory_cache: Dict[str, Any] = {}
        self.metadata: Dict[str, CacheEntry] = {}
        
        # Create cache directory structure
        if self.enable_persistent:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for category in ['novels', 'chunks', 'embeddings', 'results', 'metadata']:
                (self.cache_dir / category).mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache manager initialized at {self.cache_dir}")
        else:
            logger.info("Cache manager initialized (memory-only, no persistent storage)")
    
    def _get_cache_key(self, name: str, category: str) -> str:
        """Generate normalized cache key."""
        # Use hash for consistent file naming
        hash_input = f"{category}:{name}".encode()
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    def get_novel(self, novel_name: str) -> Optional[str]:
        """
        Retrieve cached novel text.
        
        Args:
            novel_name: Name of novel
            
        Returns:
            Novel text or None if not cached
        """
        cache_key = self._get_cache_key(novel_name, 'novel')
        
        # Try memory cache first
        if cache_key in self.memory_cache:
            logger.debug(f"Memory cache HIT for novel: {novel_name}")
            return self.memory_cache[cache_key]
        
        # Try persistent cache
        if self.enable_persistent:
            cache_file = self.cache_dir / 'novels' / f"{cache_key}.txt"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        novel_text = f.read()
                    self.memory_cache[cache_key] = novel_text
                    logger.debug(f"Disk cache HIT for novel: {novel_name}")
                    return novel_text
                except Exception as e:
                    logger.warning(f"Failed to read novel cache {cache_file}: {e}")
        
        logger.debug(f"Cache MISS for novel: {novel_name}")
        return None
    
    def cache_novel(self, novel_name: str, text: str) -> None:
        """
        Cache novel text.
        
        Args:
            novel_name: Name of novel
            text: Novel text to cache
        """
        cache_key = self._get_cache_key(novel_name, 'novel')
        
        # Store in memory
        self.memory_cache[cache_key] = text
        
        # Store on disk if persistent caching enabled
        if self.enable_persistent:
            cache_file = self.cache_dir / 'novels' / f"{cache_key}.txt"
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                logger.debug(f"Cached novel: {novel_name} ({len(text)} chars)")
            except Exception as e:
                logger.warning(f"Failed to cache novel {novel_name}: {e}")
        
        # Store metadata
        entry = CacheEntry(
            key=cache_key,
            category='novel',
            created_at=time.time()
        )
        self.metadata[cache_key] = entry
    
    def get_chunks(self, novel_name: str) -> Optional[List[Dict]]:
        """
        Retrieve cached chunks.
        
        Args:
            novel_name: Name of novel
            
        Returns:
            List of chunk dicts or None
        """
        cache_key = self._get_cache_key(novel_name, 'chunks')
        
        # Memory cache
        if cache_key in self.memory_cache:
            logger.debug(f"Memory cache HIT for chunks: {novel_name}")
            return self.memory_cache[cache_key]
        
        # Persistent cache
        if self.enable_persistent:
            cache_file = self.cache_dir / 'chunks' / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        chunks = pickle.load(f)
                    self.memory_cache[cache_key] = chunks
                    logger.debug(f"Disk cache HIT for chunks: {novel_name} ({len(chunks)} chunks)")
                    return chunks
                except Exception as e:
                    logger.warning(f"Failed to read chunks cache {cache_file}: {e}")
        
        logger.debug(f"Cache MISS for chunks: {novel_name}")
        return None
    
    def cache_chunks(self, novel_name: str, chunks: List[Dict]) -> None:
        """
        Cache narrative chunks.
        
        Args:
            novel_name: Name of novel
            chunks: List of chunk objects/dicts
        """
        cache_key = self._get_cache_key(novel_name, 'chunks')
        
        # Memory cache
        self.memory_cache[cache_key] = chunks
        
        # Persistent cache
        if self.enable_persistent:
            cache_file = self.cache_dir / 'chunks' / f"{cache_key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(chunks, f)
                logger.debug(f"Cached {len(chunks)} chunks for: {novel_name}")
            except Exception as e:
                logger.warning(f"Failed to cache chunks for {novel_name}: {e}")
        
        # Metadata
        entry = CacheEntry(
            key=cache_key,
            category='chunks',
            created_at=time.time()
        )
        self.metadata[cache_key] = entry
    
    def get_embeddings(self, novel_name: str) -> Optional[Dict[int, List[float]]]:
        """
        Retrieve cached embeddings.
        
        Args:
            novel_name: Name of novel
            
        Returns:
            Dict mapping chunk_id to embeddings or None
        """
        cache_key = self._get_cache_key(novel_name, 'embeddings')
        
        if cache_key in self.memory_cache:
            logger.debug(f"Memory cache HIT for embeddings: {novel_name}")
            return self.memory_cache[cache_key]
        
        if self.enable_persistent:
            cache_file = self.cache_dir / 'embeddings' / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        embeddings = pickle.load(f)
                    self.memory_cache[cache_key] = embeddings
                    logger.debug(f"Disk cache HIT for embeddings: {novel_name}")
                    return embeddings
                except Exception as e:
                    logger.warning(f"Failed to read embeddings cache: {e}")
        
        logger.debug(f"Cache MISS for embeddings: {novel_name}")
        return None
    
    def cache_embeddings(self, novel_name: str, embeddings: Dict[int, List[float]]) -> None:
        """
        Cache chunk embeddings.
        
        Args:
            novel_name: Name of novel
            embeddings: Dict mapping chunk_id to embedding vectors
        """
        cache_key = self._get_cache_key(novel_name, 'embeddings')
        
        self.memory_cache[cache_key] = embeddings
        
        if self.enable_persistent:
            cache_file = self.cache_dir / 'embeddings' / f"{cache_key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(embeddings, f)
                logger.debug(f"Cached embeddings for: {novel_name}")
            except Exception as e:
                logger.warning(f"Failed to cache embeddings for {novel_name}: {e}")
        
        entry = CacheEntry(
            key=cache_key,
            category='embeddings',
            created_at=time.time()
        )
        self.metadata[cache_key] = entry
    
    def get_result(self, example_id: str) -> Optional[Dict]:
        """
        Retrieve cached consistency check result.
        
        Args:
            example_id: Example identifier
            
        Returns:
            Result dict or None
        """
        cache_key = self._get_cache_key(example_id, 'result')
        
        if cache_key in self.memory_cache:
            logger.debug(f"Memory cache HIT for result: {example_id}")
            return self.memory_cache[cache_key]
        
        if self.enable_persistent:
            cache_file = self.cache_dir / 'results' / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        result = json.load(f)
                    self.memory_cache[cache_key] = result
                    logger.debug(f"Disk cache HIT for result: {example_id}")
                    return result
                except Exception as e:
                    logger.warning(f"Failed to read result cache: {e}")
        
        return None
    
    def cache_result(self, example_id: str, result: Dict) -> None:
        """
        Cache consistency check result.
        
        Args:
            example_id: Example identifier
            result: Result dict to cache
        """
        cache_key = self._get_cache_key(example_id, 'result')
        
        self.memory_cache[cache_key] = result
        
        if self.enable_persistent:
            cache_file = self.cache_dir / 'results' / f"{cache_key}.json"
            try:
                with open(cache_file, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.debug(f"Cached result for example: {example_id}")
            except Exception as e:
                logger.warning(f"Failed to cache result for {example_id}: {e}")
        
        entry = CacheEntry(
            key=cache_key,
            category='result',
            created_at=time.time()
        )
        self.metadata[cache_key] = entry
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for reporting."""
        categories = {}
        for key, entry in self.metadata.items():
            cat = entry.category
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        
        return {
            'memory_entries': len(self.memory_cache),
            'metadata_entries': len(self.metadata),
            'by_category': categories,
            'cache_dir': str(self.cache_dir),
            'persistent_enabled': self.enable_persistent
        }
    
    def clear_cache(self, category: Optional[str] = None) -> None:
        """
        Clear cache entries.
        
        Args:
            category: Specific category to clear, or None for all
        """
        if category:
            to_delete = [k for k, v in self.metadata.items() if v.category == category]
            for k in to_delete:
                if k in self.memory_cache:
                    del self.memory_cache[k]
                del self.metadata[k]
            logger.info(f"Cleared {len(to_delete)} entries from category: {category}")
        else:
            self.memory_cache.clear()
            self.metadata.clear()
            logger.info("Cleared entire cache")
    
    def print_stats(self) -> None:
        """Print cache statistics."""
        stats = self.get_cache_stats()
        logger.info("="*60)
        logger.info("CACHE STATISTICS")
        logger.info("="*60)
        logger.info(f"Memory entries: {stats['memory_entries']}")
        logger.info(f"Metadata entries: {stats['metadata_entries']}")
        logger.info(f"Persistent: {stats['persistent_enabled']}")
        if stats['by_category']:
            logger.info("By category:")
            for cat, count in stats['by_category'].items():
                logger.info(f"  {cat}: {count}")
        logger.info("="*60)
