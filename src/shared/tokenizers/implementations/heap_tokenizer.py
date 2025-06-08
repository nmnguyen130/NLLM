"""Heap-based BPE tokenizer implementation.

This module provides a BPE tokenizer that uses a heap for efficient pair counting
and a linked list for efficient merging during training.
"""

import heapq
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, DefaultDict, Any

from ..core.base import Tokenizer
from ..utils.text_processing import get_stats, merge


class TokenNode:
    """Node in a doubly-linked list for efficient BPE merging."""
    
    __slots__ = ['token_id', 'prev', 'next', 'alive']
    
    def __init__(self, token_id: int):
        self.token_id = token_id
        self.prev: Optional['TokenNode'] = None
        self.next: Optional['TokenNode'] = None
        self.alive: bool = True  # Marks if the node is still active (not merged)
    
    def __hash__(self) -> int:
        return id(self)
    
    def __eq__(self, other: Any) -> bool:
        return self is other


class HeapTokenizer(Tokenizer):
    """BPE tokenizer using a heap and linked list for efficient training.
    
    This implementation is optimized for training efficiency by using:
    - A heap to track the most frequent pairs
    - A linked list for efficient merging of tokens
    - Lazy cleanup of merged nodes
    """
    
    def _init_linked_list(self, token_ids: List[int]) -> List[TokenNode]:
        """Convert a list of token IDs into a doubly-linked list of TokenNodes."""
        nodes = [TokenNode(tid) for tid in token_ids]
        for i in range(1, len(nodes)):
            nodes[i-1].next = nodes[i]
            nodes[i].prev = nodes[i-1]
        return nodes
    
    def _get_pair_counts(
        self, 
        nodes: List[TokenNode]
    ) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], Set[TokenNode]]]:
        """Count frequencies of consecutive token pairs and track their locations."""
        counts: Dict[Tuple[int, int], int] = defaultdict(int)
        locations: DefaultDict[Tuple[int, int], Set[TokenNode]] = defaultdict(set)
        
        for node in nodes:
            if node.alive and node.next and node.next.alive:
                pair = (node.token_id, node.next.token_id)
                counts[pair] += 1
                locations[pair].add(node)
                
        return dict(counts), locations
    
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train the tokenizer on the given text.
        
        Args:
            text: Training corpus as a string
            vocab_size: Desired vocabulary size (must be >= 256)
            verbose: Whether to print training progress debug info
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")
        
        # Initialize with byte-level tokens
        text_bytes = text.encode("utf-8")
        token_ids = list(text_bytes)
        nodes = self._init_linked_list(token_ids)
        
        # Initialize pair counts and locations
        pair_counts, pair_locs = self._get_pair_counts(nodes)
        
        # Initialize heap with negative counts for min-heap behavior
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)
        
        next_token_id = 256
        num_merges = vocab_size - 256
        
        for _ in range(num_merges):
            # Find the most frequent valid pair
            while heap:
                neg_count, pair = heapq.heappop(heap)
                if pair in pair_counts and pair_counts[pair] == -neg_count:
                    break
            else:
                break  # No more valid pairs - Heap exhausted
            
            if verbose:
                print(f"\n--- Merging pair {pair} -> {next_token_id} ---")
                print("pair_counts BEFORE:", dict(pair_counts))
                print("pair_locs BEFORE:", {k: [id(n) for n in v] for k, v in pair_locs.items()})

            # Merge the most frequent pair
            p1, p2 = pair
            new_id = next_token_id
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[p1] + self.vocab[p2]
            next_token_id += 1
            
            # Track nodes that were merged
            changed_nodes = []
            
            for node in list(pair_locs[pair]):
                if not (node.alive and node.next and node.next.alive):
                    continue
                
                # Create a new merged node
                merged = TokenNode(new_id)
                left = node.prev
                right = node.next.next
                
                # Update linked list pointers
                if left:
                    left.next = merged
                    merged.prev = left
                if right:
                    right.prev = merged
                    merged.next = right
                
                # Mark original nodes as merged
                node.alive = False
                node.next.alive = False
                changed_nodes.append(merged)
                
                # Update counts for old pairs
                old_pairs = []
                if left and left.alive:
                    old_pairs.append(((left.token_id, node.token_id), left))
                if right and right.alive:
                    old_pairs.append(((node.next.token_id, right.token_id), node.next))
                
                for old_pair, old_node in old_pairs:
                    if old_pair in pair_counts:
                        pair_counts[old_pair] -= 1
                        pair_locs[old_pair].discard(old_node)
                        if pair_counts[old_pair] == 0:
                            del pair_counts[old_pair]
                            del pair_locs[old_pair]
                
            # Remove the merged pair
            if pair in pair_counts:
                del pair_counts[pair]
            if pair in pair_locs:
                del pair_locs[pair]
            
            # Update counts for new pairs formed by the merged nodes
            for merged in changed_nodes:
                # Check left side
                if merged.prev and merged.prev.alive:
                    new_pair = (merged.prev.token_id, merged.token_id)
                    if merged not in pair_locs[new_pair]:
                        pair_counts[new_pair] = pair_counts.get(new_pair, 0) + 1
                        pair_locs[new_pair].add(merged.prev)
                        heapq.heappush(heap, (-pair_counts[new_pair], new_pair))
                
                # Check right side
                if merged.next and merged.next.alive:
                    new_pair = (merged.token_id, merged.next.token_id)
                    if merged not in pair_locs[new_pair]:
                        pair_counts[new_pair] = pair_counts.get(new_pair, 0) + 1
                        pair_locs[new_pair].add(merged)
                        heapq.heappush(heap, (-pair_counts[new_pair], new_pair))

            if verbose:
                print("Remaining pairs after merge:")
                for k, v in pair_counts.items():
                    print(k, ":", v)

            if verbose:
                print(f"\n--- Merging nodes {[id(n) for n in changed_nodes]} ---")
                print("pair_counts AFTER:", dict(pair_counts))
                print("pair_locs AFTER:", {k: [id(n) for n in v] for k, v in pair_locs.items()})
                print(f"✅ Merged {pair} -> {new_id}, new heap size: {len(heap)}, heap: {heap}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text using the trained tokenizer."""
        token_ids = list(text.encode("utf-8"))
        
        while True:
            pairs = {}
            get_stats(token_ids, pairs)
            
            # Find the pair with the lowest merge index
            best_pair = None
            best_rank = float('inf')
            
            for pair in pairs:
                if pair in self.merges and self.merges[pair] < best_rank:
                    best_rank = self.merges[pair]
                    best_pair = pair
            
            if best_pair is None:
                break
                
            token_ids = merge(token_ids, best_pair, self.merges[best_pair])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to a string."""
        text_bytes = b"".join(self.vocab[tid] for tid in token_ids)
        return text_bytes.decode("utf-8", errors="replace")
