import unittest
from eole.predict.greedy_search import GreedySearch, sample_with_temperature

import torch


class GlobalScorerStub(object):
    alpha = 0
    beta = 0

    def __init__(self):
        self.length_penalty = lambda x, alpha: 1.0
        self.cov_penalty = lambda cov, beta: torch.zeros((1, cov.shape[-2]), device=cov.device, dtype=torch.float)
        self.has_cov_pen = False
        self.has_len_pen = False

    def update_global_state(self, beam):
        pass

    def score(self, beam, scores):
        return scores


class TestGreedySearch(unittest.TestCase):
    BATCH_SZ = 3
    INP_SEQ_LEN = 53
    DEAD_SCORE = -1e20

    BLOCKED_SCORE = -10e20

    def test_doesnt_predict_eos_if_shorter_than_min_len(self):
        # batch 0 will always predict EOS. The other batches will predict
        # non-eos scores.
        for batch_sz in [1, 3]:
            n_words = 100
            _non_eos_idxs = [47]
            valid_score_dist = torch.log_softmax(torch.tensor([6.0, 5.0]), dim=0)
            min_length = 5
            eos_idx = 2
            lengths = torch.randint(0, 30, (batch_sz,))
            samp = GreedySearch(
                0,
                1,
                2,
                3,
                1,
                1,
                batch_sz,
                GlobalScorerStub(),
                min_length,
                False,
                set(),
                False,
                30,
                1.0,
                1,
                0,
                1,
                False,
            )
            samp.initialize(torch.zeros((1, 1)), lengths)
            all_attns = []
            for i in range(min_length + 4):
                word_probs = torch.full((batch_sz, n_words), -float("inf"))
                # "best" prediction is eos - that should be blocked
                word_probs[0, eos_idx] = valid_score_dist[0]
                # include at least one prediction OTHER than EOS
                # that is greater than -1e20
                word_probs[0, _non_eos_idxs[0]] = valid_score_dist[1]
                word_probs[1:, _non_eos_idxs[0] + i] = 0

                attns = torch.randn(1, batch_sz, 53)
                all_attns.append(attns)
                samp.advance(word_probs, attns)
                if i < min_length:
                    self.assertTrue(samp.topk_scores[0].allclose(valid_score_dist[1]))
                    self.assertTrue(samp.topk_scores[1:].eq(0).all())
                elif i == min_length:
                    # now batch 0 has ended and no others have
                    self.assertTrue(all(samp.is_finished_list[0][:]))
                    self.assertTrue(all(all([not x for x in sublist]) for sublist in samp.is_finished_list[1:][1:]))
                else:  # i > min_length
                    break

    def test_returns_correct_scores_deterministic(self):
        for batch_sz in [1, 13]:
            for temp in [1.0, 3.0]:
                n_words = 100
                _non_eos_idxs = [47, 51, 13, 88, 99]
                valid_score_dist_1 = torch.log_softmax(torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), dim=0)
                valid_score_dist_2 = torch.log_softmax(torch.tensor([6.0, 1.0]), dim=0)
                eos_idx = 2
                lengths = torch.randint(0, 30, (batch_sz,))
                samp = GreedySearch(
                    0,
                    1,
                    2,
                    3,
                    1,
                    1,
                    batch_sz,
                    GlobalScorerStub(),
                    0,
                    False,
                    set(),
                    False,
                    30,
                    temp,
                    1,
                    0,
                    1,
                    False,
                )
                samp.initialize(torch.zeros((1, 1)), lengths)
                # initial step
                i = 0
                word_probs = torch.full((batch_sz, n_words), -float("inf"))
                # batch 0 dies on step 0
                word_probs[0, eos_idx] = valid_score_dist_1[0]
                # include at least one prediction OTHER than EOS
                # that is greater than -1e20
                word_probs[0, _non_eos_idxs] = valid_score_dist_1[1:]
                word_probs[1:, _non_eos_idxs[0] + i] = 0

                attns = torch.randn(1, batch_sz, 53)
                samp.advance(word_probs, attns)
                self.assertTrue(all(samp.is_finished_list[0]))
                samp.update_finished()
                self.assertEqual(
                    [score for score, *_ in samp.hypotheses[0]],
                    [valid_score_dist_1[0]],
                )
                if batch_sz == 1:
                    self.assertTrue(samp.done)
                    continue
                else:
                    self.assertFalse(samp.done)

                # step 2
                i = 1
                word_probs = torch.full((batch_sz - 1, n_words), -float("inf"))
                # (old) batch 8 dies on step 1
                word_probs[7, eos_idx] = valid_score_dist_2[0]
                word_probs[0:7, _non_eos_idxs[:2]] = valid_score_dist_2
                word_probs[8:, _non_eos_idxs[:2]] = valid_score_dist_2

                attns = torch.randn(1, batch_sz, 53)
                samp.advance(word_probs, attns)

                self.assertTrue(all(samp.is_finished_list[7]))
                samp.update_finished()
                self.assertEqual(
                    [score for score, *_ in samp.hypotheses[8]],
                    [valid_score_dist_2[0]],
                )

                # step 3
                i = 2
                word_probs = torch.full((batch_sz - 2, n_words), -float("inf"))
                # everything dies
                word_probs[:, eos_idx] = 0

                attns = torch.randn(1, batch_sz, 53)
                samp.advance(word_probs, attns)

                self.assertTrue(all(all(sublist) for sublist in samp.is_finished_list))
                samp.update_finished()
                self.assertTrue(samp.done)

    def test_returns_correct_scores_non_deterministic(self):
        for batch_sz in [1, 13]:
            for temp in [1.0, 3.0]:
                n_words = 100
                _non_eos_idxs = [47, 51, 13, 88, 99]
                valid_score_dist_1 = torch.log_softmax(torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), dim=0)
                valid_score_dist_2 = torch.log_softmax(torch.tensor([6.0, 1.0]), dim=0)
                eos_idx = 2
                lengths = torch.randint(0, 30, (batch_sz,))
                samp = GreedySearch(
                    0,
                    1,
                    2,
                    3,
                    1,
                    1,
                    batch_sz,
                    GlobalScorerStub(),
                    0,
                    False,
                    set(),
                    False,
                    30,
                    temp,
                    2,
                    0,
                    1,
                    False,
                )
                samp.initialize(torch.zeros((1, 1)), lengths)
                # initial step
                i = 0
                for _ in range(100):
                    word_probs = torch.full((batch_sz, n_words), -float("inf"))
                    # batch 0 dies on step 0
                    word_probs[0, eos_idx] = valid_score_dist_1[0]
                    # include at least one prediction OTHER than EOS
                    # that is greater than -1e20
                    word_probs[0, _non_eos_idxs] = valid_score_dist_1[1:]
                    word_probs[1:, _non_eos_idxs[0] + i] = 0

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if all(samp.is_finished_list[0]):
                        break
                else:
                    self.fail(
                        "Batch 0 never ended (very unlikely but maybe "
                        "due to stochasticisty. If so, please increase "
                        "the range of the for-loop."
                    )
                samp.update_finished()
                self.assertEqual([samp.topk_scores[0]], [valid_score_dist_1[0] / temp])
                if batch_sz == 1:
                    self.assertTrue(samp.done)
                    continue
                else:
                    self.assertFalse(samp.done)

                # step 2
                i = 1
                for _ in range(100):
                    word_probs = torch.full((batch_sz - 1, n_words), -float("inf"))
                    # (old) batch 8 dies on step 1
                    word_probs[7, eos_idx] = valid_score_dist_2[0]
                    word_probs[0:7, _non_eos_idxs[:2]] = valid_score_dist_2
                    word_probs[8:, _non_eos_idxs[:2]] = valid_score_dist_2

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if all(samp.is_finished_list[7]):
                        break
                else:
                    self.fail(
                        "Batch 8 never ended (very unlikely but maybe "
                        "due to stochasticisty. If so, please increase "
                        "the range of the for-loop."
                    )

                samp.update_finished()
                self.assertEqual(
                    [score for score, *_ in samp.hypotheses[8]],
                    [valid_score_dist_2[0]],
                )

                # step 3
                i = 2
                for _ in range(250):
                    word_probs = torch.full((samp.alive_seq.shape[0], n_words), -float("inf"))
                    # everything dies
                    word_probs[:, eos_idx] = 0

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if any(any(sublist) for sublist in samp.is_finished_list):
                        samp.update_finished()
                    if all(all(sublist) for sublist in samp.is_finished_list):
                        break
                else:
                    self.fail(
                        "All batches never ended (very unlikely but "
                        "maybe due to stochasticisty. If so, please "
                        "increase the range of the for-loop."
                    )

                self.assertTrue(samp.done)

    def test_returns_correct_scores_non_deterministic_beams(self):
        beam_size = 10
        for batch_sz in [1, 13]:
            for temp in [1.0, 3.0]:
                print(batch_sz, temp)
                n_words = 100
                _non_eos_idxs = [47, 51, 13, 88, 99]
                valid_score_dist_1 = torch.log_softmax(torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), dim=0)
                valid_score_dist_2 = torch.log_softmax(torch.tensor([6.0, 1.0]), dim=0)
                eos_idx = 2
                lengths = torch.randint(0, 30, (batch_sz,))
                samp = GreedySearch(
                    0,
                    1,
                    2,
                    3,
                    1,
                    1,
                    batch_sz,
                    GlobalScorerStub(),
                    0,
                    False,
                    set(),
                    False,
                    30,
                    temp,
                    50,
                    0,
                    beam_size,
                    False,
                )
                samp.initialize(torch.zeros((1, 1)), lengths)
                # initial step
                # finish one beam
                i = 0
                for _ in range(100):
                    word_probs = torch.full((batch_sz * beam_size, n_words), -float("inf"))

                    word_probs[beam_size - 2, eos_idx] = valid_score_dist_1[0]
                    # include at least one prediction OTHER than EOS
                    # that is greater than -1e20
                    word_probs[beam_size - 2, _non_eos_idxs] = valid_score_dist_1[1:]
                    word_probs[beam_size - 2 + 1 :, _non_eos_idxs[0] + i] = 0
                    word_probs[: beam_size - 2, _non_eos_idxs[0] + i] = 0

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if all(samp.is_finished_list[beam_size - 2]):
                        self.assertFalse(any(any(sublist) for sublist in samp.is_finished_list[: beam_size - 2]))
                        self.assertFalse(any(samp.is_finished_list[beam_size - 2 + 1]))
                        break
                else:
                    self.fail(
                        "Batch 0 never ended (very unlikely but maybe "
                        "due to stochasticisty. If so, please increase "
                        "the range of the for-loop."
                    )
                samp.update_finished()
                self.assertEqual([samp.topk_scores[beam_size - 2]], [valid_score_dist_1[0] / temp])

                # step 2
                # finish example in last batch
                i = 1
                for _ in range(100):
                    word_probs = torch.full((batch_sz * beam_size - 1, n_words), -float("inf"))
                    # (old) batch 8 dies on step 1
                    word_probs[(batch_sz - 1) * beam_size + 7, eos_idx] = valid_score_dist_2[0]
                    word_probs[: (batch_sz - 1) * beam_size + 7, _non_eos_idxs[:2]] = valid_score_dist_2
                    word_probs[(batch_sz - 1) * beam_size + 8 :, _non_eos_idxs[:2]] = valid_score_dist_2

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if all(samp.is_finished_list[(batch_sz - 1) * beam_size + 7]):
                        break
                else:
                    self.fail(
                        "Batch 8 never ended (very unlikely but maybe "
                        "due to stochasticisty. If so, please increase "
                        "the range of the for-loop."
                    )

                samp.update_finished()
                self.assertEqual(
                    [score for score, *_ in samp.hypotheses[batch_sz - 1][-1:]],
                    [valid_score_dist_2[0]],
                )

                # step 3
                i = 2
                for _ in range(250):
                    word_probs = torch.full((samp.alive_seq.shape[0], n_words), -float("inf"))
                    # everything dies
                    word_probs[:, eos_idx] = 0

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if any(any(sublist) for sublist in samp.is_finished_list):
                        samp.update_finished()
                    if all(all(sublist) for sublist in samp.is_finished_list):
                        break
                else:
                    self.fail(
                        "All batches never ended (very unlikely but "
                        "maybe due to stochasticisty. If so, please "
                        "increase the range of the for-loop."
                    )

                self.assertTrue(samp.done)

    def test_returns_correct_scores_non_deterministic_topp(self):
        for batch_sz in [1, 13]:
            for temp in [1.0, 0.3]:
                n_words = 100
                _non_eos_idxs = [47, 51, 13, 88, 99]
                valid_score_dist_1 = torch.log_softmax(torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), dim=0)
                valid_score_dist_2 = torch.log_softmax(torch.tensor([6.0, 1.0]), dim=0)
                eos_idx = 2
                lengths = torch.randint(0, 30, (batch_sz,))
                samp = GreedySearch(
                    0,
                    1,
                    2,
                    3,
                    1,
                    1,
                    batch_sz,
                    GlobalScorerStub(),
                    0,
                    False,
                    set(),
                    False,
                    -1,
                    temp,
                    50,
                    0.5,
                    1,
                    False,
                )
                samp.initialize(torch.zeros((1, 1)), lengths)
                # initial step
                i = 0
                for _ in range(100):
                    word_probs = torch.full((batch_sz, n_words), -float("inf"))
                    # batch 0 dies on step 0
                    word_probs[0, eos_idx] = valid_score_dist_1[0]
                    # include at least one prediction OTHER than EOS
                    # that is greater than -1e20
                    word_probs[0, _non_eos_idxs] = valid_score_dist_1[1:]
                    word_probs[1:, _non_eos_idxs[0] + i] = 0

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if all(samp.is_finished_list[0]):
                        break
                else:
                    self.fail(
                        "Batch 0 never ended (very unlikely but maybe "
                        "due to stochasticisty. If so, please increase "
                        "the range of the for-loop."
                    )
                samp.update_finished()
                self.assertEqual(
                    [score for score, *_ in samp.hypotheses[0]],
                    [valid_score_dist_1[0]],
                )
                if batch_sz == 1:
                    self.assertTrue(samp.done)
                    continue
                else:
                    self.assertFalse(samp.done)

                # step 2
                i = 1
                for _ in range(200):
                    word_probs = torch.full((batch_sz - 1, n_words), -float("inf"))
                    # (old) batch 8 dies on step 1
                    word_probs[7, eos_idx] = valid_score_dist_2[0]
                    word_probs[0:7, _non_eos_idxs[:2]] = valid_score_dist_2
                    word_probs[8:, _non_eos_idxs[:2]] = valid_score_dist_2

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if all(samp.is_finished_list[7]):
                        break
                else:
                    self.fail(
                        "Batch 8 never ended (very unlikely but maybe "
                        "due to stochasticisty. If so, please increase "
                        "the range of the for-loop."
                    )

                samp.update_finished()
                self.assertEqual(
                    [score for score, *_ in samp.hypotheses[8]],
                    [valid_score_dist_2[0]],
                )

                # step 3
                i = 2
                for _ in range(250):
                    word_probs = torch.full((samp.alive_seq.shape[0], n_words), -float("inf"))
                    # everything dies
                    word_probs[:, eos_idx] = 0

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if any(any(sublist) for sublist in samp.is_finished_list):
                        samp.update_finished()
                    if all(all(sublist) for sublist in samp.is_finished_list):
                        break
                else:
                    self.fail(
                        "All batches never ended (very unlikely but "
                        "maybe due to stochasticisty. If so, please "
                        "increase the range of the for-loop."
                    )

                self.assertTrue(samp.done)


class TestSampleWithTemperatureNanInf(unittest.TestCase):
    """Test that sample_with_temperature handles NaN/Inf logits gracefully.

    With torch.compile (especially max-autotune mode), compiled CUDA kernels
    can emit NaN or Inf values in the logit tensor due to BF16 overflow or
    numerical instability in fused attention kernels.  If these are not
    sanitised before sampling, torch.distributions.Categorical internally
    calls torch.multinomial with an all-NaN probability tensor which triggers
    a CUDA device-side assertion:
        "probability tensor contains either `inf`, `nan` or element < 0".
    """

    def test_nan_in_logits_does_not_crash(self):
        """NaN logits are replaced with a large negative value so that the
        corresponding tokens are excluded from sampling."""
        logits = torch.randn(2, 100)
        logits[0, 5] = float("nan")
        logits[1, :50] = float("nan")  # half the vocab is NaN for row 1
        topk_ids, topk_scores = sample_with_temperature(logits, 1.0, 0, 0.9)
        self.assertEqual(topk_ids.shape, (2, 1))
        # Row 1: must choose from the valid half (indices 50-99)
        self.assertGreaterEqual(topk_ids[1, 0].item(), 50)

    def test_posinf_in_logits_chooses_that_token(self):
        """+Inf logits are replaced with a large positive value so that they
        still dominate the distribution."""
        logits = torch.full((2, 100), -1e9)
        logits[0, 42] = float("inf")
        logits[1, 7] = float("inf")
        topk_ids, _ = sample_with_temperature(logits, 1.0, 0, 1.0)
        self.assertEqual(topk_ids[0, 0].item(), 42)
        self.assertEqual(topk_ids[1, 0].item(), 7)

    def test_neginf_logits_still_sample_from_valid_token(self):
        """-Inf logits (legitimate masking) are replaced with a large negative
        value, preserving the intent: the one valid token should win."""
        logits = torch.full((1, 100), float("-inf"))
        logits[0, 33] = 0.0  # sole valid token
        topk_ids, _ = sample_with_temperature(logits, 1.0, 0, 0.9)
        self.assertEqual(topk_ids[0, 0].item(), 33)

    def test_greedy_argmax_unaffected_by_fix(self):
        """temperature=0 (argmax) branch does NOT apply nan_to_num and
        should behave exactly as before."""
        logits = torch.tensor([[1.0, 3.0, 2.0]])
        topk_ids, _ = sample_with_temperature(logits, 0.0, 0, 0.9)
        self.assertEqual(topk_ids[0, 0].item(), 1)  # argmax = index 1
