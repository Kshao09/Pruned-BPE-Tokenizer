from typing import List, Optional, Tuple

def find_best_pair(corpus: List[List[int]]) -> Optional[Tuple[int, int, int]]: ...

def merge_corpus(
    corpus: List[List[int]],
    left_id: int,
    right_id: int,
    new_id: int,
) -> List[List[int]]: ...