from typing import List, Dict, Any
import torch
import torch.nn as nn


class LinearPerceptron(nn.Module):
    """Linear Perceptron Baseline.
    
    Demonstrates the theoretical ceiling of linear decision boundaries on non-convex manifolds.
    Maps: R^2 -> R^1 (logits)
    """

    def __init__(self, input_dim: int = 2, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        probs = self.predict_proba(x)
        return (probs >= threshold).long()


class ShallowMLP(nn.Module):
    """Shallow Multi-Layer Perceptron.
    
    Proof-of-concept non-linear coordinate warping.
    Architecture: R^2 -> R^16 -> Activation -> R^1
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (self.predict_proba(x) >= threshold).long()


class DeepModularMLP(nn.Module):
    """Deep Multi-Layer Perceptron with Modular Representation Learning.

    Architecture features:
    - Dynamically parameterized hidden dimensions [d1, d2, ..., dk]
    - LayerNormalization or BatchNormalization
    - Configurable activations: GELU, LeakyReLU, ReLU, Tanh
    - Dropout regularization for structural generalization
    - Numerically stable raw logit output
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: List[int] = None,
        output_dim: int = 1,
        activation: str = "gelu",
        dropout_rate: float = 0.10,
        use_layer_norm: bool = True
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(self._get_activation(activation))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h_dim

        # Final projection layer to scalar logit
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights(activation)

    def _get_activation(self, name: str) -> nn.Module:
        name_lower = name.lower()
        if name_lower == "relu":
            return nn.ReLU()
        elif name_lower == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)
        elif name_lower == "gelu":
            return nn.GELU()
        elif name_lower == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: '{name}'. Choose from ['relu', 'leaky_relu', 'gelu', 'tanh'].")

    def _init_weights(self, activation: str):
        nonlinearity = "relu" if activation in ["relu", "leaky_relu", "gelu"] else "linear"
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity=nonlinearity)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (self.predict_proba(x) >= threshold).long()


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Instantiate model based on declarative configuration dictionary."""
    model_cfg = config["model"]
    arch = model_cfg.get("architecture", "DeepModularMLP")

    if arch == "LinearPerceptron":
        return LinearPerceptron(
            input_dim=model_cfg.get("input_dim", 2),
            output_dim=model_cfg.get("output_dim", 1)
        )
    elif arch == "ShallowMLP":
        return ShallowMLP(
            input_dim=model_cfg.get("input_dim", 2),
            hidden_dim=model_cfg.get("hidden_dims", [16])[0],
            output_dim=model_cfg.get("output_dim", 1)
        )
    elif arch == "DeepModularMLP":
        return DeepModularMLP(
            input_dim=model_cfg.get("input_dim", 2),
            hidden_dims=model_cfg.get("hidden_dims", [64, 32, 16]),
            output_dim=model_cfg.get("output_dim", 1),
            activation=model_cfg.get("activation", "gelu"),
            dropout_rate=model_cfg.get("dropout_rate", 0.10),
            use_layer_norm=model_cfg.get("use_layer_norm", True)
        )
    else:
        raise ValueError(f"Unknown model architecture requested: '{arch}'")
