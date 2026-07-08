"""
PrunedBPETrainerCythonParallel with multiprocessing support.

This keeps the Cython pair-counting / merge functions in bpe_fast_core_parallel.pyx,
but uses multiple long-lived Python worker processes so a large CPU can be used.

Put this beside bpe_fast_core_parallel.pyx after building the extension.
"""
import os
import traceback
import math
import multiprocessing as mp
from pathlib import Path
from collections import Counter
from typing import List, Optional, Dict, Any

from bpe_fast_core_parallel import find_pair_stats, merge_corpus_inplace, unpack_pair
from pruned_bpe_pretokenizer import pretokenize
from PrunedBPETrainer import PrunedBPETrainer, NUM_RESERVED_TOKENS

_WORKER_CMD_FIND = "find_pair_stats"
_WORKER_CMD_MERGE = "merge"
_WORKER_CMD_GET_CORPUS = "get_corpus"
_WORKER_CMD_CLOSE = "close"
_WORKER_OK = "ok"
_WORKER_ERROR = "error"


def _make_contiguous_shards(corpus: List[List[int]], num_shards: int) -> List[List[List[int]]]:
    """
    Split corpus into contiguous shards.

    Contiguous order is important because the trainer uses first_seen as a
    deterministic tie-breaker when two pairs have the same count.
    """
    if num_shards <= 1 or len(corpus) <= 1:
        return [corpus]

    pair_counts = [max(0, len(ids) - 1) for ids in corpus]
    total_pairs = sum(pair_counts)

    if total_pairs == 0:
        return [corpus]

    target_pairs_per_shard = max(1, total_pairs // num_shards)

    shards = []
    current = []
    current_pairs = 0

    for ids, pair_count in zip(corpus, pair_counts):
        if current and current_pairs >= target_pairs_per_shard and len(shards) < num_shards - 1:
            shards.append(current)
            current = []
            current_pairs = 0

        current.append(ids)
        current_pairs += pair_count

    if current:
        shards.append(current)

    return shards


def _parallel_bpe_worker(conn, shard):
    """
    Long-lived worker process.

    The shard stays inside this process. This is important on Windows because
    repeatedly sending the full corpus to workers would destroy performance.
    """
    try:
        while True:
            cmd, payload = conn.recv()

            if cmd == _WORKER_CMD_FIND:
                result = find_pair_stats(shard)
                conn.send((_WORKER_OK, result))
            elif cmd == _WORKER_CMD_MERGE:
                left_id, right_id, new_id = payload
                merge_corpus_inplace(shard, left_id, right_id, new_id)
                conn.send((_WORKER_OK, None))
            elif cmd == _WORKER_CMD_GET_CORPUS:
                conn.send((_WORKER_OK, shard))
            elif cmd == _WORKER_CMD_CLOSE:
                conn.send((_WORKER_OK, None))
                break
            else:
                raise ValueError(f"Unknown worker command: {cmd}")
    except BaseException:
        conn.send((_WORKER_ERROR, traceback.format_exc()))
    finally:
        conn.close()


class _ParallelCorpus:
    """
    Owns multiple worker processes, each with one corpus shard.
    """
    def __init__(self, corpus: List[List[int]], num_workers: int = 16):
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")

        # Do not create more workers than useful shards.
        self.shards = _make_contiguous_shards(corpus, num_workers)
        self.num_workers = len(self.shards)

        self.ctx = mp.get_context("spawn")
        self.processes = []
        self.conns = []
        self.closed = False

        for shard_index, shard in enumerate(self.shards):
            parent_conn, child_conn = self.ctx.Pipe(duplex=True)
            proc = self.ctx.Process(
                target=_parallel_bpe_worker,
                args=(child_conn, shard),
                name=f"BPEWorker-{shard_index}",
            )
            proc.start()
            child_conn.close()

            self.conns.append(parent_conn)
            self.processes.append(proc)

        print(f"[ParallelBPE] Started {self.num_workers} worker process(es).", flush=True)

    def _send_all(self, cmd: str, payload: Any = None) -> None:
        for conn in self.conns:
            conn.send((cmd, payload))

    def _recv_all(self) -> List[Any]:
        results = []
        for conn in self.conns:
            status, payload = conn.recv()
            if status == _WORKER_ERROR:
                raise RuntimeError(f"BPE worker failed:\n{payload}")
            results.append(payload)
        return results

    def find_best_pair(self):
        """
        Return (left_id, right_id, count), or None.
        """
        self._send_all(_WORKER_CMD_FIND)
        shard_results = self._recv_all()

        global_stats: Dict[int, List[int]] = {}
        shard_seen_offset = 0

        for stats, shard_pair_count in shard_results:
            for key, value in stats.items():
                count, first_seen_local = value
                first_seen_global = shard_seen_offset + first_seen_local

                old = global_stats.get(key)
                if old is None:
                    global_stats[key] = [count, first_seen_global]
                else:
                    old[0] += count
                    if first_seen_global < old[1]:
                        old[1] = first_seen_global

            shard_seen_offset += shard_pair_count

        if not global_stats:
            return None

        best_key = None
        best_count = -1
        best_first_seen = None

        for key, value in global_stats.items():
            count, first_seen = value

            if (
                best_key is None
                or count > best_count
                or (count == best_count and first_seen < best_first_seen)
            ):
                best_key = key
                best_count = count
                best_first_seen = first_seen

        left_id, right_id = unpack_pair(best_key)
        return left_id, right_id, int(best_count)

    def merge(self, left_id: int, right_id: int, new_id: int) -> None:
        self._send_all(_WORKER_CMD_MERGE, (left_id, right_id, new_id))
        self._recv_all()

    def get_corpus(self) -> List[List[int]]:
        self._send_all(_WORKER_CMD_GET_CORPUS)
        shards = self._recv_all()

        corpus = []
        for shard in shards:
            corpus.extend(shard)
        return corpus

    def close(self) -> None:
        if self.closed:
            return

        for conn in self.conns:
            try:
                conn.send((_WORKER_CMD_CLOSE, None))
            except (BrokenPipeError, EOFError):
                pass

        for conn in self.conns:
            try:
                status, payload = conn.recv()
                if status == _WORKER_ERROR:
                    print(f"[ParallelBPE] Worker close error:\n{payload}", flush=True)
            except (BrokenPipeError, EOFError):
                pass
            finally:
                conn.close()

        for proc in self.processes:
            proc.join()

        self.closed = True

    def terminate(self) -> None:
        if self.closed:
            return

        for conn in self.conns:
            try:
                conn.close()
            except Exception:
                pass

        for proc in self.processes:
            if proc.is_alive():
                proc.terminate()
            proc.join()

        self.closed = True


class PrunedBPETrainerCythonParallel(PrunedBPETrainer):
    def train(
            self,
            texts: Optional[List[str]] = None,
            checkpoint_vocab_sizes: Optional[List[int]] = None,
            checkpoint_dir: Optional[str] = None,
            num_workers: int = 16,
    ) -> None:
        if checkpoint_vocab_sizes is None:
            checkpoint_vocab_sizes = []

        checkpoint_vocab_sizes = set(checkpoint_vocab_sizes)

        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Fresh training.
        if texts is not None:
            corpus = []
            for text in texts:
                if not text:
                    continue

                for chunk in pretokenize(text):
                    if chunk:
                        corpus.append(self._text_to_ids(chunk))
        # Resume training.
        else:
            if not self.corpus:
                raise ValueError("texts is None, but no checkpoint corpus exists.")
            corpus = self.corpus

        parallel_corpus = _ParallelCorpus(corpus, num_workers=num_workers)

        try:
            next_id = self._next_training_id()
            while next_id < self.train_vocab_size:
                result = parallel_corpus.find_best_pair()
                if result is None:
                    break

                left_id, right_id, count = result
                if count < 2:
                    break

                best_pair = (left_id, right_id)

                self.merges[best_pair] = next_id
                self.id_to_children[next_id] = best_pair
                self.id_to_bytes[next_id] = (
                    self.id_to_bytes[left_id] + self.id_to_bytes[right_id]
                )

                if not hasattr(self, "merge_counts"):
                    self.merge_counts = {}
                self.merge_counts[next_id] = count

                parallel_corpus.merge(left_id, right_id, next_id)

                token_text_repr = self._format_token_text(self.id_to_bytes[next_id])
                print(
                    f"Merge {next_id}: {best_pair} -> {next_id}, count={count}, token={token_text_repr}",
                    flush=True
                )

                next_id += 1

                # Effective vocab size is now next_id, because IDs 0...next_id-1 exist.
                if next_id in checkpoint_vocab_sizes:
                    self.corpus = parallel_corpus.get_corpus()

                    self.final_token_counts = Counter()
                    for ids in self.corpus:
                        self.final_token_counts.update(ids)

                    if checkpoint_dir is None:
                        checkpoint_path = f"checkpoint_vocab_{next_id}.pkl"
                    else:
                        checkpoint_path = os.path.join(
                            checkpoint_dir,
                            f"checkpoint_vocab_{next_id}.pkl"
                        )

                    self.save_checkpoint(checkpoint_path)

            self.corpus = parallel_corpus.get_corpus()
        except BaseException:
            parallel_corpus.terminate()
            raise
        else:
            parallel_corpus.close()

        self.final_token_counts = Counter()
        for ids in self.corpus:
            self.final_token_counts.update(ids)


if __name__ == "__main__":
    import time
    from datetime import datetime
    from settings import PROJECT_ROOT

    corpus_dir = os.path.join(PROJECT_ROOT, "Corpus", "Corpus1")
    vocab_path = os.path.join(PROJECT_ROOT, "vocab_1.txt")
    inter_vocab_path = os.path.join(PROJECT_ROOT, "inter_vocab_1.txt")
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", "c1")

    # External target vocabulary size includes reserved special tokens.
    TARGET_VOCAB_SIZE = 10_000
    # Internal trainer visible vocabulary excludes reserved special tokens.
    VISIBLE_VOCAB_SIZE = TARGET_VOCAB_SIZE - NUM_RESERVED_TOKENS

    # Stage 1:
    # Use MIN_EXPOSURE_COUNT = 0 and train exactly to VISIBLE_VOCAB_SIZE.
    #
    # Stage 2:
    # Load the Stage 1 checkpoint, set MIN_EXPOSURE_COUNT to the pruning
    # threshold, and continue training until visible learned tokens are full.
    STAGE = 1
    if STAGE == 1:
        MIN_EXPOSURE_COUNT = 0
        TRAIN_VOCAB_SIZE = VISIBLE_VOCAB_SIZE
        CHECKPOINT_VOCAB_SIZES = [8996, TRAIN_VOCAB_SIZE]
        # RESUME_CHECKPOINT_PATH = None
        RESUME_CHECKPOINT_PATH = os.path.join(checkpoint_dir, "checkpoint_vocab_8190.pkl")
    elif STAGE == 2:
        # Look up the frequency of the last token trained in stage 1, and then * 0.2, * 0.3, or * 0.4 here
        MIN_EXPOSURE_COUNT = math.ceil(4260 * 0.4)
        # The loaded trainer still needs a train_vocab_size setting, but in stage 2 this is only
        # the starting size. The real upper limit is MAX_TRAIN_VOCAB_SIZE below.
        TRAIN_VOCAB_SIZE = VISIBLE_VOCAB_SIZE
        # Safety bound. Make this generous enough.
        # Training will stop earlier once visible learned tokens are full.
        MAX_TRAIN_VOCAB_SIZE = math.ceil(1.2 * VISIBLE_VOCAB_SIZE)
        CHECKPOINT_VOCAB_SIZES = []
        # Load the Stage 1 checkpoint or any closer previous checkpoint.
        RESUME_CHECKPOINT_PATH = os.path.join(checkpoint_dir, "checkpoint_vocab_8131.pkl")
    else:
        raise ValueError(f"Invalid STAGE: {STAGE}")

    NUM_WORKERS = 24

    start_time = time.perf_counter()
    if RESUME_CHECKPOINT_PATH is None:
        print("Loading corpus...")
        texts = PrunedBPETrainerCythonParallel.load_data(corpus_dir)

        print(f"\nFinished loading corpus.")
        print(f"Number of text items: {len(texts)}")
        print(f"Training start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        trainer = PrunedBPETrainerCythonParallel(
            train_vocab_size=TRAIN_VOCAB_SIZE,
            visible_vocab_size=VISIBLE_VOCAB_SIZE,
            min_exposure_count=MIN_EXPOSURE_COUNT,
        )
    else:
        texts = None
        trainer = PrunedBPETrainerCythonParallel.load_checkpoint(
            RESUME_CHECKPOINT_PATH,
            train_vocab_size=TRAIN_VOCAB_SIZE,
            visible_vocab_size=VISIBLE_VOCAB_SIZE,
            min_exposure_count=MIN_EXPOSURE_COUNT,
        )

        print(f"Resume training start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Resume from next_id={trainer._next_training_id()}")
        print(f"Visible vocab size={trainer.visible_vocab_size}")
        print(f"min_exposure_count={trainer.min_exposure_count}")

    if STAGE == 1:
        trainer.train(
            texts=texts,
            checkpoint_vocab_sizes=CHECKPOINT_VOCAB_SIZES,
            checkpoint_dir=checkpoint_dir,
            num_workers=NUM_WORKERS,
        )
    else:
        trainer.train_until_visible_vocab_full(
            max_train_vocab_size=MAX_TRAIN_VOCAB_SIZE,
            checkpoint_vocab_sizes=CHECKPOINT_VOCAB_SIZES,
            checkpoint_dir=checkpoint_dir,
            step_size=25,
            num_workers=NUM_WORKERS,
        )

    train_elapsed = time.perf_counter() - start_time

    print(f"\nTraining finished.")
    print(f"Training elapsed seconds: {train_elapsed:.2f}")
    print(f"Training elapsed hours: {train_elapsed / 3600:.4f}")

    trainer.save_vocab(vocab_path, inter_vocab_path)