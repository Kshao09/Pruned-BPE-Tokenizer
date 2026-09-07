from typing import Dict, List, Tuple

def find_pair_stats(
    corpus: List[List[int]],
) -> Tuple[Dict[int, Tuple[int, int]], int]: ...

def merge_corpus_inplace(
    corpus: List[List[int]],
    left_id: int,
    right_id: int,
    new_id: int,
) -> None: ...

def unpack_pair(key: int) -> Tuple[int, int]: ...
