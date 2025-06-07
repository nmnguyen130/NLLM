import heapq
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set
import tqdm
from .base import Tokenizer, get_stats, merge, render_token

# === DATA STRUCTURES - DOUBLE LINKED LIST ===

class TokenNode:
    def __init__(self, token_id: int):
        self.token_id = token_id
        self.prev: Optional['TokenNode'] = None
        self.next: Optional['TokenNode'] = None
        self.alive: bool = True  # Used to mark nodes that have been merged

    def __hash__(self):
        return id(self)
    
    def __eq__(self, other):
        return self is other

# === UTILITY ===

def pair_key(a: TokenNode, b: TokenNode) -> Tuple[int, int]:
    return (a.token_id, b.token_id)

# === BPE Tokenizer ===

class BPETokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.final_tokens: List[int] = []

    def _init_linked_list(self, token_ids: List[int]) -> List[TokenNode]:
        """
        Turn a list of token IDs into a doubly linked list of TokenNodes.
        """
        nodes = [TokenNode(tid) for tid in token_ids]
        for i in range(1, len(nodes)):
            nodes[i-1].next = nodes[i]
            nodes[i].prev = nodes[i-1]
        return nodes

    def _get_pair_counts(self, nodes: List[TokenNode]) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], Set[TokenNode]]]:
        """
        Count the frequency of consecutive token pairs in the LinkedList `nodes`.
        Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        """
        counts = defaultdict(int)
        locations = defaultdict(set)
        for node in nodes:
            if node.alive and node.next and node.next.alive:
                k = pair_key(node, node.next)
                counts[k] += 1
                locations[k].add(node)
        return counts, locations

    def train(self, text: str, vocab_size: int, verbose=False):
        assert vocab_size >= 256, "vocab_size must be >= 256"
        
        text_bytes = text.encode("utf-8")
        token_ids = list(text_bytes)
        nodes = self._init_linked_list(token_ids)
        pair_counts, pair_locs = self._get_pair_counts(nodes)

        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        next_token_id = 256
        num_merges = vocab_size - 256

        for _ in tqdm.trange(num_merges, desc="Training tokenizer"):
            while heap:
                neg_count, pair = heapq.heappop(heap)
                if pair in pair_counts and pair_counts[pair] == -neg_count:
                    break
            else:
                break  # Heap exhausted

            if verbose:
                print(f"\n--- Merging pair {pair} -> {next_token_id} ---")
                print("pair_counts BEFORE:", dict(pair_counts))
                print("pair_locs BEFORE:", {k: [id(n) for n in v] for k, v in pair_locs.items()})

            p1, p2 = pair
            new_id = next_token_id
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[p1] + self.vocab[p2]
            next_token_id += 1

            changed_nodes = []
            for node in list(pair_locs[pair]):
                if not (node.alive and node.next and node.next.alive):
                    continue

                # Merge the pair into new node
                merged = TokenNode(new_id)
                left = node.prev
                right = node.next.next

                if left:
                    left.next = merged
                    merged.prev = left
                if right:
                    right.prev = merged
                    merged.next = right

                node.alive = False
                node.next.alive = False
                changed_nodes.append(merged)

                old_pairs = []
                if left and left.alive:
                    old_pairs.append((pair_key(left, node), left))
                if right and right.alive:
                    old_pairs.append((pair_key(node.next, right), node.next))

                for old_pair, old_node in old_pairs:
                    if old_pair in pair_counts:
                        pair_counts[old_pair] -= 1
                        pair_locs[old_pair].discard(old_node)
                        if pair_counts[old_pair] == 0:
                            del pair_counts[old_pair]
                            del pair_locs[old_pair]

            del pair_counts[pair]
            del pair_locs[pair]

            # Update pair counts locally from merged nodes
            for merged in changed_nodes:
                if merged.prev and merged.prev.alive:
                    new_pair = pair_key(merged.prev, merged)
                    if merged.prev not in pair_locs[new_pair]:
                        pair_counts[new_pair] += 1
                        pair_locs[new_pair].add(merged.prev)
                        heapq.heappush(heap, (-pair_counts[new_pair], new_pair))
                
                if merged.next and merged.next.alive:
                    new_pair = pair_key(merged, merged.next)
                    if merged not in pair_locs[new_pair]:
                        pair_counts[new_pair] += 1
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
            
        # Bước 11: Lưu danh sách token cuối cùng
        self.final_tokens = [node.token_id for node in nodes if node.alive]

    def encode(self, text: str) -> List[int]:
        token_ids = list(text.encode("utf-8"))
        while True:
            pairs = get_stats(token_ids)
            candidate = None
            best_rank = float('inf')
            for pair in pairs:
                if pair in self.merges and self.merges[pair] < best_rank:
                    best_rank = self.merges[pair]
                    candidate = pair
            if candidate is None:
                break
            token_ids = merge(token_ids, candidate, self.merges[candidate])
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        text_bytes = b"".join(self.vocab[t] for t in token_ids)
        return text_bytes.decode("utf-8", errors="replace")
    
if __name__ == "__main__":
    tokenizer = BPETokenizer()
    # with open("data/taylorswift.txt", encoding="utf-8") as f:
    #     text = f.read()

    text = "the theater is there for those who think they're thinkers"

    tokenizer.train(text, vocab_size=1000, verbose=False)
    tokenizer.save("taylorswift")

    # s = "Reading practice to help you understand texts with everyday or job-related language. Texts include articles, travel guides, emails, adverts and reviews."
    s = "abcab"
    tokens = tokenizer.encode(s)
    # print("\nTokens:")
    # for token_id in tokens:
    #     token_bytes = tokenizer.vocab[token_id]
    #     try:
    #         token_str = token_bytes.decode("utf-8")
    #     except UnicodeDecodeError:
    #         token_str = str(token_bytes)
    #     print(f"{token_id}: {token_str}")
    print("Encoded:", tokens)
    print("Decoded:", tokenizer.decode(tokens))