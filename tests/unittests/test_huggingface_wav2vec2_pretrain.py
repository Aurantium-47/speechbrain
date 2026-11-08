"""Offline tests for Wav2Vec2 pretraining masks and negative samples."""

import numpy as np
import pytest
import torch


@pytest.fixture
def pretrain_model(tmp_path, device):
    transformers = pytest.importorskip("transformers")
    from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2Pretrain

    config = transformers.Wav2Vec2Config(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        conv_dim=(8, 8),
        conv_stride=(2, 2),
        conv_kernel=(3, 3),
        feat_extract_norm="layer",
        num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=2,
        num_codevector_groups=2,
        num_codevectors_per_group=4,
        codevector_dim=8,
        proj_codevector_dim=8,
        num_negatives=3,
        mask_time_length=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        feat_proj_dropout=0.0,
        final_dropout=0.0,
        layerdrop=0.0,
    )
    source = tmp_path / "model"
    transformers.Wav2Vec2ForPreTraining(config).save_pretrained(source)
    model = Wav2Vec2Pretrain(
        str(source),
        str(tmp_path / "cache"),
        mask_prob=0.65,
        mask_length=2,
        normalize_wav=False,
    ).to(device)
    return model


@pytest.mark.parametrize(
    "lengths", [[1.0, 0.375], [0.75, 0.375], [1.0, 0.0625], [1.0, 11 / 256]]
)
@pytest.mark.parametrize("normalize_wav", [False, True])
@pytest.mark.parametrize("mask_length", [2, 10])
def test_pretraining_excludes_padding(
    pretrain_model, device, lengths, normalize_wav, mask_length
):
    model = pretrain_model
    model.normalize_wav = normalize_wav
    model.mask_length = mask_length
    wavs = torch.randn(2, 256, device=device)
    wav_lens = torch.tensor(lengths, device=device)
    raw_lengths = torch.round(wav_lens * wavs.shape[1]).long()
    for wav, length in zip(wavs, raw_lengths):
        wav[length:] = 0

    inputs = {}

    def capture_inputs(module, args, kwargs):
        inputs.update(kwargs)

    hook = model.model.register_forward_pre_hook(
        capture_inputs, with_kwargs=True
    )
    state = np.random.get_state()
    try:
        np.random.seed(42)
        outputs, mask = model(wavs, wav_lens)
    finally:
        np.random.set_state(state)
        hook.remove()

    feature_lengths = model.model._get_feat_extract_output_lengths(raw_lengths)
    sequence_length = mask.shape[1]
    negatives = inputs["sampled_negative_indices"]
    for batch_index, length in enumerate(feature_lengths):
        if length >= mask_length:
            assert mask[batch_index, :length].any()
        assert not mask[batch_index, length:].any()
        local_negatives = negatives[batch_index, :length] - (
            batch_index * sequence_length
        )
        assert (local_negatives >= 0).all()
        assert (local_negatives < length).all()
        targets = torch.arange(length, device=device).unsqueeze(-1)
        assert (local_negatives != targets).all()

    assert torch.isfinite(outputs.loss)
    outputs.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(grad).all() for grad in gradients)


def test_pretraining_full_lengths_match_omitted_lengths(pretrain_model, device):
    model = pretrain_model.eval()
    wavs = torch.randn(2, 256, device=device)
    state = np.random.get_state()
    try:
        np.random.seed(42)
        without_lengths, mask_without = model(wavs)
        np.random.seed(42)
        with_lengths, mask_with = model(wavs, torch.ones(2, device=device))
    finally:
        np.random.set_state(state)

    torch.testing.assert_close(mask_without, mask_with)
    torch.testing.assert_close(without_lengths.loss, with_lengths.loss)
    torch.testing.assert_close(
        without_lengths.projected_states, with_lengths.projected_states
    )


@pytest.mark.parametrize("short_length", [0, 3, 7])
def test_pretraining_rejects_too_few_valid_frames(
    pretrain_model, device, short_length
):
    wavs = torch.randn(2, 256, device=device)
    lengths = torch.tensor([1.0, short_length / 256], device=device)
    with pytest.raises(ValueError, match="at least two feature frames"):
        pretrain_model(wavs, lengths)


def test_pretraining_rejects_too_few_unpadded_frames(pretrain_model, device):
    with pytest.raises(ValueError, match="at least two feature frames"):
        pretrain_model(torch.randn(2, 7, device=device))


def test_pretraining_rejects_empty_target_mask(pretrain_model, device):
    pretrain_model.mask_length = 10
    wavs = torch.randn(2, 256, device=device)
    lengths = torch.full((2,), 16 / 256, device=device)
    with pytest.raises(ValueError, match="No valid feature frames"):
        pretrain_model(wavs, lengths)
