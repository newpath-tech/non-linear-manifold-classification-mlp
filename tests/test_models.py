import pytest
import torch
from src.models.architectures import LinearPerceptron, ShallowMLP, DeepModularMLP, build_model


def test_linear_perceptron_shapes():
    model = LinearPerceptron(input_dim=2, output_dim=1)
    x = torch.randn(32, 2)
    logits = model(x)
    probs = model.predict_proba(x)
    preds = model.predict(x)

    assert logits.shape == (32, 1)
    assert probs.shape == (32, 1)
    assert preds.shape == (32, 1)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))
    assert set(preds.unique().tolist()).issubset({0, 1})


def test_shallow_mlp():
    model = ShallowMLP(input_dim=2, hidden_dim=16, output_dim=1)
    x = torch.randn(16, 2)
    logits = model(x)
    probs = model.predict_proba(x)
    assert logits.shape == (16, 1)
    assert probs.shape == (16, 1)


@pytest.mark.parametrize("act", ["relu", "leaky_relu", "gelu", "tanh"])
def test_deep_modular_mlp_activations_and_gradients(act):
    model = DeepModularMLP(
        input_dim=2,
        hidden_dims=[32, 16],
        output_dim=1,
        activation=act,
        dropout_rate=0.1,
        use_layer_norm=True
    )
    x = torch.randn(16, 2, requires_grad=True)
    logits = model(x)
    assert logits.shape == (16, 1)

    # Verify backpropagation gradient flow
    loss = logits.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_gpu_device_allocation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available on host system.")

    device = torch.device("cuda:0")
    model = DeepModularMLP().to(device)
    x = torch.randn(64, 2, device=device)

    logits = model(x)
    assert logits.device.type == "cuda"
    assert logits.shape == (64, 1)
