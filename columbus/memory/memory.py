"""
Enhanced memory layer with Mem0 and Qdrant integration
"""

import os
import asyncio
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from mem0 import Memory
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition
import numpy as np
from columbus.config.settings import Config


class MemoryEntry(BaseModel):
    id: str
    text: str
    user_id: str
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[float] = None
    embedding: Optional[List[float]] = None
    relevance_score: Optional[float] = None


class MemoryQuery(BaseModel):
    query: str
    user_id: str = "default"
    tags: Optional[List[str]] = None
    limit: int = 5
    min_relevance: float = 0.7
    include_metadata: bool = True


class MemoryStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # Memory configuration
        self.mem0_api_key = os.getenv("MEM0_API_KEY")
        self.qdrant_url = getattr(cfg, 'qdrant_url', 'http://localhost:6333')
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = getattr(cfg, 'memory_collection', 'columbus_memory')
        
        # Initialize Mem0 for structured memory management
        self.mem0_client = self._init_mem0()
        
        # Initialize Qdrant for vector storage and retrieval
        self.qdrant_client = self._init_qdrant()
        
        # Setup vector collection
        self._setup_vector_collection()
    
    def _init_mem0(self) -> Optional[Memory]:
        """Initialize Mem0 client with Qdrant backend"""
        try:
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": f"{self.collection_name}_mem0",
                        "url": self.qdrant_url,
                        "api_key": self.qdrant_api_key,
                    }
                },
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": getattr(self.cfg, 'memory_model', 'llama3.2:3b'),
                        "ollama_base_url": getattr(self.cfg, 'ollama_base_url', 'http://localhost:11434')
                    }
                }
            }
            
            # Add OpenAI config if API key is available
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                config["embedder"] = {
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-3-small"
                    }
                }
            
            return Memory.from_config(config)
        except Exception as e:
            print(f"Warning: Mem0 initialization failed: {e}")
            print("Note: Memory features disabled. Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
            return None
    
    def _init_qdrant(self) -> Optional[QdrantClient]:
        """Initialize Qdrant client"""
        try:
            if self.qdrant_url.startswith("http"):
                return QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key
                )
            else:
                # Local file-based storage
                return QdrantClient(path=self.qdrant_url)
        except Exception as e:
            print(f"Warning: Qdrant initialization failed: {e}")
            return None
    
    def _setup_vector_collection(self):
        """Setup Qdrant collection for vector storage"""
        if not self.qdrant_client:
            return
        
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if not collection_exists:
                # Create collection with appropriate vector size
                # Using 1536 for OpenAI embeddings, adjust if using different embeddings
                vector_size = 1536  # text-embedding-3-small dimension
                
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            print(f"Warning: Failed to setup Qdrant collection: {e}")
    
    def remember(self, text: str, user_id: str = "default", tags: Optional[List[str]] = None, 
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a memory using both Mem0 and Qdrant"""
        try:
            # Use Mem0 for structured memory storage
            if self.mem0_client:
                mem0_metadata = {
                    "tags": tags or [],
                    "user_id": user_id,
                    **(metadata or {})
                }
                
                self.mem0_client.add(
                    text, 
                    user_id=user_id, 
                    metadata=mem0_metadata
                )
            
            # Also store in Qdrant for direct vector search capabilities
            if self.qdrant_client:
                self._store_in_qdrant(text, user_id, tags, metadata)
            
            return True
            
        except Exception as e:
            print(f"Error storing memory: {e}")
            return False
    
    def _store_in_qdrant(self, text: str, user_id: str, tags: Optional[List[str]], 
                        metadata: Optional[Dict[str, Any]]):
        """Store memory directly in Qdrant"""
        try:
            # Generate a simple embedding (in production, use proper embedding model)
            # This is a placeholder - you'd use OpenAI embeddings or similar
            import hashlib
            point_id = hashlib.md5(f"{user_id}:{text}".encode()).hexdigest()
            
            # Create a simple embedding based on text hash (placeholder)
            embedding = self._generate_simple_embedding(text)
            
            import time
            
            payload = {
                "text": text,
                "user_id": user_id,
                "tags": tags or [],
                "metadata": metadata or {},
                "timestamp": time.time()
            }
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )]
            )
            
        except Exception as e:
            print(f"Error storing in Qdrant: {e}")
    
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """Generate a simple embedding (placeholder for real embedding model)"""
        # This is a very basic embedding - replace with proper embedding model
        import hashlib
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to float vector (1536 dimensions for OpenAI compatibility)
        embedding = []
        for i in range(0, len(hash_bytes), 4):
            chunk = hash_bytes[i:i+4]
            if len(chunk) == 4:
                val = int.from_bytes(chunk, 'big') / (2**32 - 1)  # Normalize to 0-1
                embedding.append(val * 2 - 1)  # Convert to -1 to 1 range
        
        # Pad or truncate to 1536 dimensions
        while len(embedding) < 1536:
            embedding.extend(embedding[:min(len(embedding), 1536 - len(embedding))])
        
        return embedding[:1536]
    
    def recall(self, query: str, user_id: str = "default", limit: int = 5, 
               tags: Optional[List[str]] = None) -> List[MemoryEntry]:
        """Retrieve relevant memories using Mem0 and Qdrant"""
        memories = []
        
        try:
            # Use Mem0 for semantic memory search
            if self.mem0_client:
                mem0_results = self.mem0_client.search(
                    query,
                    user_id=user_id,
                    limit=limit
                )
                
                for result in mem0_results:
                    memory = MemoryEntry(
                        id=str(result.get('id', '')),
                        text=result.get('text', ''),
                        user_id=user_id,
                        tags=result.get('metadata', {}).get('tags', []),
                        metadata=result.get('metadata', {}),
                        relevance_score=result.get('score', 0.0)
                    )
                    memories.append(memory)
            
            # Supplement with direct Qdrant search if needed
            if self.qdrant_client and len(memories) < limit:
                qdrant_memories = self._search_qdrant(query, user_id, limit - len(memories), tags)
                memories.extend(qdrant_memories)
            
        except Exception as e:
            print(f"Error recalling memories: {e}")
        
        return memories[:limit]
    
    def _search_qdrant(self, query: str, user_id: str, limit: int, 
                      tags: Optional[List[str]]) -> List[MemoryEntry]:
        """Search directly in Qdrant"""
        try:
            query_embedding = self._generate_simple_embedding(query)
            
            # Build filter conditions
            filter_conditions = [
                FieldCondition(key="user_id", match={"value": user_id})
            ]
            
            if tags:
                for tag in tags:
                    filter_conditions.append(
                        FieldCondition(key="tags", match={"any": [tag]})
                    )
            
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=Filter(must=filter_conditions) if filter_conditions else None,
                limit=limit,
                with_payload=True
            )
            
            memories = []
            for hit in search_result:
                payload = hit.payload
                memory = MemoryEntry(
                    id=str(hit.id),
                    text=payload.get('text', ''),
                    user_id=payload.get('user_id', user_id),
                    tags=payload.get('tags', []),
                    metadata=payload.get('metadata', {}),
                    timestamp=payload.get('timestamp'),
                    relevance_score=hit.score
                )
                memories.append(memory)
            
            return memories
            
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return []
    
    def get_user_memories(self, user_id: str, limit: int = 50) -> List[MemoryEntry]:
        """Get all memories for a specific user"""
        if not self.qdrant_client:
            return []
        
        try:
            # Get all memories for user without vector search
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="user_id", match={"value": user_id})]
                ),
                limit=limit,
                with_payload=True
            )
            
            memories = []
            for point in scroll_result[0]:
                payload = point.payload
                memory = MemoryEntry(
                    id=str(point.id),
                    text=payload.get('text', ''),
                    user_id=payload.get('user_id', user_id),
                    tags=payload.get('tags', []),
                    metadata=payload.get('metadata', {}),
                    timestamp=payload.get('timestamp')
                )
                memories.append(memory)
            
            return memories
            
        except Exception as e:
            print(f"Error getting user memories: {e}")
            return []
    
    def delete_memory(self, memory_id: str, user_id: str = "default") -> bool:
        """Delete a specific memory"""
        try:
            # Delete from Mem0 if available
            if self.mem0_client:
                # Mem0 delete by memory_id (this may vary based on Mem0 API)
                # Check Mem0 documentation for exact delete method
                pass
            
            # Delete from Qdrant
            if self.qdrant_client:
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=[memory_id]
                )
            
            return True
            
        except Exception as e:
            print(f"Error deleting memory: {e}")
            return False
    
    def clear_user_memories(self, user_id: str) -> bool:
        """Clear all memories for a specific user"""
        try:
            # Clear from Qdrant
            if self.qdrant_client:
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=Filter(
                        must=[FieldCondition(key="user_id", match={"value": user_id})]
                    )
                )
            
            return True
            
        except Exception as e:
            print(f"Error clearing user memories: {e}")
            return False
