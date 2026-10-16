"""Torch-free tests of the Integrated Gradients kernel, word aggregation, cache and schemas.

Written with unittest so they run with plain ``python -m unittest`` as well as under pytest.
"""
import unittest

import numpy as np

from app.ai import integrated_gradients as ig


class KernelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.normal(size=(5, 4))
        self.b = rng.normal(size=(5, 4))

    def test_quadratic_matches_closed_form(self):
        w = np.arange(1, 21, dtype=float).reshape(5, 4) / 10

        def fn(p):  # F = sum(w * x^2); grad = 2 w x  (linear in alpha -> trapezoid is exact)
            return (w * p ** 2).sum(axis=(1, 2)), 2 * w * p

        r = ig.integrated_gradients(self.x, self.b, fn, steps=8)
        expected = (w * (self.x ** 2 - self.b ** 2)).sum(axis=1)
        np.testing.assert_allclose(r.token_attributions, expected, rtol=1e-9, atol=1e-9)
        self.assertTrue(r.converged)
        self.assertLess(r.completeness_error, 1e-9)

    def test_completeness_and_convergence_on_nonlinear_function(self):
        def fn(p):  # F = sum(tanh(x))
            return np.tanh(p).sum(axis=(1, 2)), 1 - np.tanh(p) ** 2

        errs = [ig.integrated_gradients(self.x, self.b, fn, steps=s).completeness_error for s in (2, 8, 32)]
        self.assertLess(errs[2], errs[0])
        self.assertTrue(ig.integrated_gradients(self.x, self.b, fn, steps=32).converged)

    def test_signs_are_preserved(self):
        x = np.array([[1.0], [1.0]])
        b = np.zeros((2, 1))
        sign = np.array([[1.0], [-1.0]])

        def fn(p):  # position 0 increases F, position 1 decreases it
            return (sign * p).sum(axis=(1, 2)), np.broadcast_to(sign, p.shape).copy()

        r = ig.integrated_gradients(x, b, fn, steps=4)
        self.assertGreater(r.token_attributions[0], 0)
        self.assertLess(r.token_attributions[1], 0)

    def test_identical_input_and_baseline_gives_zero(self):
        fn = lambda p: (np.tanh(p).sum(axis=(1, 2)), 1 - np.tanh(p) ** 2)  # noqa: E731
        r = ig.integrated_gradients(self.x, self.x, fn, steps=4)
        np.testing.assert_array_equal(r.token_attributions, 0)
        self.assertIsNone(r.relative_error)
        self.assertTrue(r.converged)

    def test_batching_does_not_change_result(self):
        fn = lambda p: (np.tanh(p).sum(axis=(1, 2)), 1 - np.tanh(p) ** 2)  # noqa: E731
        a = ig.integrated_gradients(self.x, self.b, fn, steps=20, batch_size=3)
        b = ig.integrated_gradients(self.x, self.b, fn, steps=20, batch_size=64)
        np.testing.assert_allclose(a.token_attributions, b.token_attributions, rtol=1e-10)

    def test_non_finite_values_raise(self):
        def fn(p):
            g = np.ones_like(p)
            g[0, 0, 0] = np.nan
            return np.zeros(p.shape[0]), g

        with self.assertRaises(ig.AttributionFailed):
            ig.integrated_gradients(self.x, self.b, fn, steps=4)

    def test_bad_arguments(self):
        fn = lambda p: (np.zeros(p.shape[0]), np.zeros_like(p))  # noqa: E731
        with self.assertRaises(ValueError):
            ig.integrated_gradients(self.x, self.b[:3], fn)
        with self.assertRaises(ValueError):
            ig.integrated_gradients(self.x, self.b, fn, steps=0)
        with self.assertRaises(ValueError):
            ig.integrated_gradients(self.x, self.b, lambda p: (np.zeros(1), np.zeros_like(p)), steps=4)

    def test_output_shape(self):
        fn = lambda p: (np.tanh(p).sum(axis=(1, 2)), 1 - np.tanh(p) ** 2)  # noqa: E731
        self.assertEqual(ig.integrated_gradients(self.x, self.b, fn, steps=4).token_attributions.shape, (5,))


class AggregationTests(unittest.TestCase):
    # "Deadlocks, unhappiness!" with WordPiece-like splitting
    TEXT = "Deadlocks, unhappiness!"
    TOKENS = ["[CLS]", "dead", "##lock", "##s", ",", "un", "##hap", "##pi", "##ness", "!", "[SEP]"]
    OFFSETS = [(0, 0), (0, 4), (4, 8), (8, 9), (9, 10), (11, 13), (13, 16), (16, 18), (18, 22), (22, 23), (0, 0)]
    SPECIAL = [True] + [False] * 9 + [True]
    ATTR = [0.0, 0.10, 0.05, -0.02, -0.01, 0.03, 0.02, -0.04, 0.06, -0.005, 0.0]

    def words(self, attr=None):
        return ig.aggregate_words(self.TEXT, self.TOKENS, self.OFFSETS, self.SPECIAL, attr or self.ATTR)

    def test_subwords_merge_and_punctuation_separate(self):
        w = self.words()
        self.assertEqual([x.word for x in w], ["Deadlocks", ",", "unhappiness", "!"])
        self.assertAlmostEqual(w[0].attribution, 0.13)
        self.assertEqual([t for t, _ in w[0].tokens], ["dead", "##lock", "##s"])
        self.assertAlmostEqual(w[2].attribution, 0.07)

    def test_sum_is_preserved_and_special_tokens_skipped(self):
        w = self.words()
        self.assertAlmostEqual(sum(x.attribution for x in w), sum(a for a, s in zip(self.ATTR, self.SPECIAL) if not s))
        self.assertNotIn("[CLS]", [t for x in w for t, _ in x.tokens])

    def test_negative_sign_kept_and_normalised_range(self):
        w = self.words()
        self.assertLess(w[1].attribution, 0)
        self.assertLess(w[1].normalized, 0)
        self.assertTrue(all(-1 <= x.normalized <= 1 for x in w))
        self.assertAlmostEqual(max(abs(x.normalized) for x in w), 1.0)

    def test_truncation_drops_trailing_words_only(self):
        w = ig.aggregate_words(self.TEXT, self.TOKENS[:5] + ["[SEP]"], self.OFFSETS[:5] + [(0, 0)],
                               self.SPECIAL[:5] + [True], self.ATTR[:5] + [0.0])
        self.assertEqual([x.word for x in w], ["Deadlocks", ","])

    def test_empty_and_all_zero(self):
        self.assertEqual(ig.aggregate_words("", ["[CLS]", "[SEP]"], [(0, 0), (0, 0)], [True, True], [0, 0]), [])
        self.assertEqual(ig.normalize_scores([0.0, 0.0]), [0.0, 0.0])
        self.assertEqual(ig.normalize_scores([]), [])


class CacheTests(unittest.TestCase):
    def test_key_changes_with_every_input(self):
        base = ig.cache_key("m", "sentence", "context", 32)
        self.assertEqual(base, ig.cache_key("m", "sentence", "context", 32))
        for other in (ig.cache_key("m2", "sentence", "context", 32), ig.cache_key("m", "sentence2", "context", 32),
                      ig.cache_key("m", "sentence", "context2", 32), ig.cache_key("m", "sentence", "context", 16)):
            self.assertNotEqual(base, other)
        self.assertNotIn("sentence", base)  # digest only: no raw text

    def test_lru_eviction(self):
        c = ig.AttributionCache(max_entries=2)
        c.put("a", 1); c.put("b", 2); c.get("a"); c.put("c", 3)
        self.assertEqual((c.get("a"), c.get("b"), c.get("c")), (1, None, 3))
        self.assertEqual(len(c), 2)


class PoolingDetectionTests(unittest.TestCase):
    """The same model can describe its pooling in different ways depending on the library release."""

    class P:  # tiny stand-in with arbitrary attributes
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def test_boolean_attributes_only(self):
        modes, _ = ig.pooling_modes(self.P(pooling_mode_mean_tokens=True, pooling_mode_cls_token=False, pooling_mode_max_tokens=False))
        self.assertEqual(modes, {"mean"})

    def test_pooling_mode_string_only(self):
        self.assertEqual(ig.pooling_modes(self.P(pooling_mode="mean"))[0], {"mean"})
        self.assertEqual(ig.pooling_modes(self.P(pooling_mode="mean_tokens"))[0], {"mean"})

    def test_getter_method(self):
        class G:
            def get_pooling_mode_str(self):
                return "mean"
        self.assertEqual(ig.pooling_modes(G())[0], {"mean"})

    def test_non_mean_pooling_is_not_mean(self):
        self.assertEqual(ig.pooling_modes(self.P(pooling_mode_cls_token=True, pooling_mode_mean_tokens=False))[0], {"cls"})
        self.assertEqual(ig.pooling_modes(self.P(pooling_mode="cls+mean"))[0], {"cls", "mean"})

    def test_undeterminable_reports_none_with_diagnostic(self):
        modes, diag = ig.pooling_modes(self.P(something_else=1))
        self.assertIsNone(modes)
        self.assertIn("no pooling attributes found", diag)

    def _embedder(self, pooling, *extra):
        tok = type("Tok", (), {"is_fast": True, "pad_token_id": 0})()
        transformer = self.P(tokenizer=tok, auto_model=object(), max_seq_length=256)
        return self.P(st_model=[transformer, pooling, *extra])

    def test_components_accepts_mean_with_attribute_variants(self):
        for pooling in (self.P(pooling_mode_mean_tokens=True), self.P(pooling_mode="mean")):
            _, _, max_len = ig._components(self._embedder(pooling))
            self.assertEqual(max_len, ig.MAX_TOKENS)

    def test_components_accepts_optional_normalize_layer(self):
        class Normalize:
            pass
        ig._components(self._embedder(self.P(pooling_mode="mean"), Normalize()))

    def test_components_rejects_with_informative_messages(self):
        class Dense:
            pass
        cases = [
            (self.P(pooling_mode_cls_token=True, pooling_mode_mean_tokens=False), r"cls"),
            (self.P(unknown=1), r"Could not determine the pooling mode"),
        ]
        for pooling, pattern in cases:
            with self.assertRaises(ig.AttributionUnavailable) as cm:
                ig._components(self._embedder(pooling))
            self.assertEqual(cm.exception.reason, "unsupported_model")
            self.assertRegex(cm.exception.message, pattern)
        with self.assertRaises(ig.AttributionUnavailable) as cm:  # a Dense layer changes the embedding
            ig._components(self._embedder(self.P(pooling_mode="mean"), Dense()))
        self.assertIn("Dense", cm.exception.message)


class UnavailableTests(unittest.TestCase):
    def test_lexical_like_embedder_is_unsupported(self):
        class Lex:  # no st_model attribute
            pass

        with self.assertRaises(ig.AttributionUnavailable) as cm:
            ig.explain(Lex(), "some text", np.ones(4))
        self.assertEqual(cm.exception.reason, "unsupported_backend")

    def test_empty_input(self):
        with self.assertRaises(ig.AttributionUnavailable) as cm:
            ig.explain(object(), "   ", np.ones(4))
        self.assertEqual(cm.exception.reason, "empty_input")

    def test_broken_model_structure(self):
        class Broken:
            st_model = object()  # not subscriptable

        with self.assertRaises(ig.AttributionUnavailable) as cm:
            ig.explain(Broken(), "text", np.ones(4))
        self.assertEqual(cm.exception.reason, "unsupported_model")


if __name__ == "__main__":
    unittest.main()
