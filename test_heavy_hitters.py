import random
import unittest
from collections import Counter

from heavy_hitters import (
    STATUS_CONFIRMED,
    STATUS_POSSIBLE,
    HeavyHitterSketch,
)


def true_counts(stream):
    counts = Counter()
    for key, weight in stream:
        counts[key] += weight
    return counts


class TestUniformTraffic(unittest.TestCase):
    def test_uniform_traffic_estimates_within_bound(self):
        rng = random.Random(7)
        keys = [f"key-{i}" for i in range(200)]
        stream = [(rng.choice(keys), 1) for _ in range(20_000)]
        sketch = HeavyHitterSketch(width=1024, depth=5, capacity=512)
        for key, weight in stream:
            sketch.add(key, weight)

        counts = true_counts(stream)
        err = sketch.error_bound
        self.assertAlmostEqual(sketch.total, 20_000)
        for key, true_count in counts.items():
            est = sketch.estimate(key)
            self.assertGreaterEqual(est, true_count)
            self.assertLessEqual(est - true_count, err)

        top = sketch.topk(10)
        self.assertEqual(len(top), 10)
        estimates = [item.estimate for item in top]
        self.assertEqual(estimates, sorted(estimates, reverse=True))
        for item in top:
            self.assertGreaterEqual(item.estimate, counts[item.key])
            self.assertLessEqual(item.estimate - counts[item.key], err)
            self.assertEqual(item.upper, item.estimate)
            self.assertAlmostEqual(item.lower, max(0.0, item.estimate - err))


class TestSingleKeySkew(unittest.TestCase):
    def test_skewed_key_is_confirmed_top1(self):
        rng = random.Random(11)
        sketch = HeavyHitterSketch(width=512, depth=5, capacity=256)
        for _ in range(90_000):
            sketch.add("hot", 1)
        for _ in range(10_000):
            sketch.add(f"cold-{rng.randrange(500)}", 1)

        top = sketch.topk(5)
        self.assertEqual(top[0].key, "hot")
        self.assertEqual(top[0].status, STATUS_CONFIRMED)
        self.assertGreaterEqual(top[0].estimate, 90_000)
        self.assertLessEqual(top[0].estimate - 90_000, sketch.error_bound)
        self.assertGreaterEqual(top[0].lower, 90_000 - sketch.error_bound)

        hitters = sketch.heavy_hitters(phi=0.5)
        self.assertEqual([item.key for item in hitters], ["hot"])
        self.assertEqual(hitters[0].status, STATUS_CONFIRMED)


class TestRepeatedKeys(unittest.TestCase):
    def test_repeated_keys_and_weights(self):
        sketch = HeavyHitterSketch(width=256, depth=4, capacity=64)
        for _ in range(1_000):
            sketch.add("a", 2)
            sketch.add("a", 1)
        sketch.add("b", 500)
        self.assertEqual(sketch.total, 3_500)
        self.assertGreaterEqual(sketch.estimate("a"), 3_000)
        self.assertLessEqual(sketch.estimate("a") - 3_000, sketch.error_bound)
        self.assertGreaterEqual(sketch.estimate("b"), 500)

    def test_invalid_input_rejected(self):
        sketch = HeavyHitterSketch(width=8, depth=2, capacity=4)
        with self.assertRaises(ValueError):
            sketch.add("x", -1)
        with self.assertRaises(TypeError):
            sketch.add(b"bytes-key", 1)


class TestShardMerge(unittest.TestCase):
    def _stream(self, n=30_000):
        rng = random.Random(23)
        return [(f"key-{rng.randrange(1_000)}", rng.randrange(1, 5)) for _ in range(n)]

    def test_merged_shards_match_single_sketch(self):
        stream = self._stream()
        shards = [HeavyHitterSketch(width=512, depth=5, capacity=1_000) for _ in range(4)]
        for i, (key, weight) in enumerate(stream):
            shards[i % 4].add(key, weight)

        # Simulate cross-process transport: serialize, deserialize, then merge.
        transported = [HeavyHitterSketch.loads(shard.dumps()) for shard in shards]
        merged = HeavyHitterSketch.merged(transported)

        single = HeavyHitterSketch(width=512, depth=5, capacity=1_000)
        for key, weight in stream:
            single.add(key, weight)

        self.assertEqual(merged.dumps(), single.dumps())
        counts = true_counts(stream)
        for key, true_count in counts.items():
            self.assertGreaterEqual(merged.estimate(key), true_count)
            self.assertLessEqual(
                merged.estimate(key) - true_count, merged.error_bound
            )

    def test_merge_order_is_irrelevant(self):
        stream = self._stream(10_000)
        shards = [HeavyHitterSketch(width=256, depth=4, capacity=1_000) for _ in range(3)]
        for i, (key, weight) in enumerate(stream):
            shards[i % 3].add(key, weight)
        forward = HeavyHitterSketch.merged(shards)
        backward = HeavyHitterSketch.merged(list(reversed(shards)))
        self.assertEqual(forward.dumps(), backward.dumps())

    def test_shape_mismatch_rejected(self):
        a = HeavyHitterSketch(width=64, depth=3, capacity=8)
        b = HeavyHitterSketch(width=32, depth=3, capacity=8)
        with self.assertRaises(ValueError):
            a.merge(b)


class TestEmptyInput(unittest.TestCase):
    def test_empty_sketch(self):
        sketch = HeavyHitterSketch(width=64, depth=3, capacity=8)
        self.assertEqual(sketch.total, 0)
        self.assertEqual(sketch.error_bound, 0)
        self.assertEqual(sketch.estimate("anything"), 0)
        self.assertEqual(sketch.topk(10), [])
        self.assertEqual(sketch.heavy_hitters(0.01), [])

    def test_merge_with_empty(self):
        empty = HeavyHitterSketch(width=64, depth=3, capacity=8)
        other = HeavyHitterSketch(width=64, depth=3, capacity=8)
        other.add("x", 3)
        empty.merge(other)
        self.assertEqual(empty.estimate("x"), 3)
        self.assertEqual(empty.topk(1)[0].key, "x")


class TestTinyBudget(unittest.TestCase):
    def test_tiny_budget_still_bounded(self):
        sketch = HeavyHitterSketch.from_budget(budget_bytes=64)
        rng = random.Random(3)
        stream = [(f"k{rng.randrange(50)}", 1) for _ in range(5_000)]
        for key, weight in stream:
            sketch.add(key, weight)
        counts = true_counts(stream)
        err = sketch.error_bound
        for key, true_count in counts.items():
            est = sketch.estimate(key)
            self.assertGreaterEqual(est, true_count)
            self.assertLessEqual(est - true_count, err)
        top = sketch.topk(3)
        self.assertLessEqual(len(top), 3)
        for item in top:
            self.assertIn(item.status, (STATUS_CONFIRMED, STATUS_POSSIBLE))

    def test_from_budget_memory_and_validation(self):
        sketch = HeavyHitterSketch.from_budget(budget_bytes=64_000, delta=0.01)
        self.assertLessEqual(sketch.memory_bytes, 64_000 + 64)
        self.assertGreater(sketch.width, 1)
        self.assertGreaterEqual(sketch.depth, 1)
        with self.assertRaises(ValueError):
            HeavyHitterSketch.from_budget(0)
        with self.assertRaises(ValueError):
            HeavyHitterSketch.from_budget(1_000, delta=1.5)


class TestOrderStability(unittest.TestCase):
    def test_same_weights_different_orders_identical_result(self):
        rng = random.Random(99)
        multiset = [(f"key-{rng.randrange(300)}", rng.randrange(1, 4)) for _ in range(15_000)]
        snapshots = []
        for seed in range(5):
            shuffled = list(multiset)
            random.Random(seed).shuffle(shuffled)
            sketch = HeavyHitterSketch(width=512, depth=5, capacity=512)
            for key, weight in shuffled:
                sketch.add(key, weight)
            snapshots.append(sketch)
        reference = snapshots[0]
        for sketch in snapshots[1:]:
            self.assertEqual(sketch.dumps(), reference.dumps())
            self.assertEqual(sketch.topk(20), reference.topk(20))


class TestSerialization(unittest.TestCase):
    def test_roundtrip_preserves_state(self):
        sketch = HeavyHitterSketch(width=128, depth=4, capacity=32)
        for i in range(2_000):
            sketch.add(f"key-{i % 100}", i % 7)
        clone = HeavyHitterSketch.loads(sketch.dumps())
        self.assertEqual(clone.dumps(), sketch.dumps())
        self.assertEqual(clone.topk(10), sketch.topk(10))
        self.assertEqual(clone.total, sketch.total)

    def test_corrupt_payload_rejected(self):
        sketch = HeavyHitterSketch(width=16, depth=2, capacity=4)
        sketch.add("a", 1)
        with self.assertRaises(Exception):
            HeavyHitterSketch.loads(b"not-json")
        with self.assertRaises(ValueError):
            HeavyHitterSketch.loads(b'{"version": 999}')


class TestConfirmedVsPossible(unittest.TestCase):
    def test_status_distinguishes_confirmed_from_possible(self):
        # Narrow sketch -> large error band, so borderline keys stay "possible".
        sketch = HeavyHitterSketch(width=16, depth=2, capacity=64)
        for _ in range(10_000):
            sketch.add("dominant", 1)
        for _ in range(2_500):
            sketch.add("borderline", 1)
        for i in range(100):
            sketch.add(f"tail-{i}", 1)

        # threshold = error_bound = epsilon * total, i.e. phi == epsilon.
        hitters = sketch.heavy_hitters(phi=sketch.epsilon)
        by_key = {item.key: item for item in hitters}
        self.assertEqual(set(by_key), {"dominant", "borderline"})

        dominant = by_key["dominant"]
        self.assertEqual(dominant.status, STATUS_CONFIRMED)
        self.assertGreaterEqual(dominant.lower, sketch.error_bound)

        # "borderline" clears the threshold on its estimate, but its worst-case
        # count (estimate - error_bound) does not: possible, not confirmed.
        borderline = by_key["borderline"]
        self.assertEqual(borderline.status, STATUS_POSSIBLE)
        self.assertGreaterEqual(borderline.estimate, sketch.error_bound)
        self.assertLess(borderline.lower, sketch.error_bound)

        # Tail keys never reach the threshold and are excluded entirely.
        self.assertNotIn("tail-0", by_key)

        top = sketch.topk(3)
        self.assertEqual(top[0].key, "dominant")
        self.assertEqual(top[0].status, STATUS_CONFIRMED)


if __name__ == "__main__":
    unittest.main()
