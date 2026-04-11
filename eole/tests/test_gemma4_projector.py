import unittest
import torch
from types import SimpleNamespace

from eole.adapters.adapters import Gemma4MultiModalProjector


def _build_projector(hidden_size=8, patch_size=2, pooling_kernel_size=2):
    model_config = SimpleNamespace()
    model_config.encoder = SimpleNamespace()
    model_config.decoder = SimpleNamespace()
    model_config.encoder.hidden_size = hidden_size
    model_config.decoder.hidden_size = hidden_size
    model_config.encoder.patch_size = patch_size
    model_config.encoder.pooling_kernel_size = pooling_kernel_size
    return Gemma4MultiModalProjector(model_config)


class TestGemma4MultiModalProjector(unittest.TestCase):
    def test_packed_multi_image_is_split_and_pooled_per_image(self):
        projector = _build_projector(hidden_size=8, patch_size=2, pooling_kernel_size=2)
        # Two images, each 4x4 px -> 2x2 patches -> 4 tokens/image before pooling.
        image_sizes = torch.tensor([[4, 4], [4, 4]])
        x = torch.randn(1, 8, 8)  # packed: (1, 8 total tokens, 8 hidden_size)

        y = projector(x, image_sizes=image_sizes)
        self.assertEqual(y.shape, (2, 1, 8))

    def test_packed_length_mismatch_raises(self):
        projector = _build_projector(hidden_size=8, patch_size=2, pooling_kernel_size=2)
        image_sizes = torch.tensor([[4, 4], [4, 4]])
        x = torch.randn(1, 7, 8)  # expected 8 pre-pool tokens

        with self.assertRaises(ValueError):
            projector(x, image_sizes=image_sizes)


if __name__ == "__main__":
    unittest.main()
