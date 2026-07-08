# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False

cdef inline unsigned long long pack_pair(Py_ssize_t left, Py_ssize_t right):
    return (<unsigned long long><unsigned int>left << 32) | <unsigned int>right


cdef inline int unpack_left(unsigned long long key):
    return <int>(key >> 32)


cdef inline int unpack_right(unsigned long long key):
    return <int>(key & 0xffffffff)


def unpack_pair(unsigned long long key):
    return unpack_left(key), unpack_right(key)


def find_pair_stats(object corpus):
    """
    Count adjacent token pairs in one corpus shard.

    Returns:
        stats: dict[key] = (count, first_seen_local)
        pair_count: total number of adjacent pairs in this shard

    Important:
        first_seen_local is local to this shard.
        The main process will add the shard offset later.
    """
    cdef dict stats = {}
    cdef Py_ssize_t si, i, n
    cdef Py_ssize_t seen_index = 0

    cdef object ids
    cdef Py_ssize_t left
    cdef Py_ssize_t right
    cdef unsigned long long key

    cdef object old
    cdef Py_ssize_t count
    cdef Py_ssize_t first_seen

    for si in range(len(corpus)):
        ids = corpus[si]
        n = len(ids)

        if n < 2:
            continue

        for i in range(n - 1):
            left = <Py_ssize_t>ids[i]
            right = <Py_ssize_t>ids[i + 1]

            key = pack_pair(left, right)
            old = stats.get(key)

            if old is None:
                stats[key] = (1, seen_index)
            else:
                count = <Py_ssize_t>old[0]
                first_seen = <Py_ssize_t>old[1]
                stats[key] = (count + 1, first_seen)

            seen_index += 1

    return stats, int(seen_index)


def merge_corpus_inplace(object corpus, int left_id, int right_id, int new_id):
    """
    Merge one pair inside one corpus shard.

    This modifies the shard in-place.
    """
    cdef Py_ssize_t si, i, n
    cdef object ids
    cdef list new_ids

    cdef Py_ssize_t cur
    cdef Py_ssize_t nxt

    for si in range(len(corpus)):
        ids = corpus[si]
        n = len(ids)

        if n < 2:
            continue

        new_ids = []
        i = 0

        while i < n:
            cur = <Py_ssize_t>ids[i]

            if i < n - 1:
                nxt = <Py_ssize_t>ids[i + 1]

                if cur == left_id and nxt == right_id:
                    new_ids.append(new_id)
                    i += 2
                    continue

            new_ids.append(cur)
            i += 1

        corpus[si] = new_ids

    return None