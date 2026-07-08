# cython: language_level=3
# cython: boundscheck=True
# cython: wraparound=True
# cython: nonecheck=True
# cython: initializedcheck=True

cdef inline unsigned long long pack_pair(Py_ssize_t left, Py_ssize_t right):
    return (<unsigned long long><unsigned int>left << 32) | <unsigned int>right


cdef inline int unpack_left(unsigned long long key):
    return <int>(key >> 32)


cdef inline int unpack_right(unsigned long long key):
    return <int>(key & 0xffffffff)


def find_best_pair(object corpus):
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

    cdef unsigned long long best_key = 0
    cdef Py_ssize_t best_count = 0
    cdef Py_ssize_t best_first_seen = 0
    cdef bint have_best = False

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

    if not stats:
        return None

    for key, old in stats.items():
        count = <Py_ssize_t>old[0]
        first_seen = <Py_ssize_t>old[1]

        if (
            not have_best or
            count > best_count or
            (count == best_count and first_seen < best_first_seen)
        ):
            have_best = True
            best_key = key
            best_count = count
            best_first_seen = first_seen

    return (unpack_left(best_key), unpack_right(best_key), int(best_count))


def merge_corpus(object corpus, int left_id, int right_id, int new_id):
    cdef Py_ssize_t si, i, n
    cdef object ids
    cdef list new_ids
    cdef list out = []

    cdef Py_ssize_t cur
    cdef Py_ssize_t nxt

    for si in range(len(corpus)):
        ids = corpus[si]
        n = len(ids)

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

        out.append(new_ids)

    return out