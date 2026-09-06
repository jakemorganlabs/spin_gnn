# the stacked model: an encoder, distinct layers, and the two readout heads.
# spec: docs/SPEC.md. Prop 3 and Prop 4 (model-level symmetry) are proved in
# docs/MATH.md.
# flow:
# 1. the IdentityEncoder seeds h, h_c, and V from the raw constellation.
# 2. per layer: message pass, satellite update, controller pool, controller update.
# 3. after all layers the heads read h_c and predicates read the rigid state.

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from spin_gnn.constellation import assert_valid
from spin_gnn.encode.identity_encoder import IdentityEncoder
from spin_gnn.layers.controller import ControllerPool
from spin_gnn.layers.message import MessageLayer
from spin_gnn.layers.update import ControllerUpdate, SatelliteUpdate
from spin_gnn.model.config import SpinGnnConfig
from spin_gnn.model.readout import ContinuousHead, DiscreteHead, Predicates, predicates
from spin_gnn.types import Constellation


@dataclass(frozen=True)
class ModelOutput:
    # the frozen product of one forward call; the caller narrows on the fields.
    constellation: Constellation
    y: Tensor  # (B, k_out) continuous readout
    z: Tensor  # (B, k_classes) classification logits
    preds: Predicates


class SpinGnnLayer(nn.Module):
    # one full round: message pass, satellite update, controller pool, controller update.
    def __init__(self, config: SpinGnnConfig) -> None:
        super().__init__()
        self.config: SpinGnnConfig = config
        self.message: MessageLayer = MessageLayer(config)
        self.update: SatelliteUpdate = SatelliteUpdate(config)
        self.pool: ControllerPool = ControllerPool(config)
        self.controller: ControllerUpdate = ControllerUpdate(config)

    def forward(self, c: Constellation, debug: bool = False) -> Constellation:
        # step 1: require a valid constellation at the layer boundary.
        assert_valid(c)
        msg = self.message(c)
        out = self.update(c, msg)
        pool = self.pool(out)
        h_c = self.controller(out.h_c, msg.m_c, pool)
        # step 2: x_c never moves; the update passes it through identically.
        assert out.x_c is c.x_c, "the satellite update must pass x_c through"
        result = Constellation(
            x=out.x,
            s=out.s,
            u=out.u,
            phi=out.phi,
            omega=out.omega,
            h=out.h,
            v=out.v,
            x_c=c.x_c,
            h_c=h_c,
        )
        assert_valid(result)
        if debug:
            # step 3: in debug every layer boundary is checked again explicitly.
            assert_valid(result)
        return result


class SpinGnn(nn.Module):
    # the four-layer model; build it through build_model so the seed is applied.
    def __init__(self, config: SpinGnnConfig, k_out: int = 3, k_classes: int = 4) -> None:
        super().__init__()
        self.config: SpinGnnConfig = config
        self.k_out: int = k_out
        self.k_classes: int = k_classes
        self.encoder: IdentityEncoder = IdentityEncoder(config)
        # step 1: distinct layers, no weight sharing.
        self.layers: nn.ModuleList = nn.ModuleList(
            SpinGnnLayer(config) for _ in range(config.n_layers)
        )
        self.continuous: ContinuousHead = ContinuousHead(config.d_c, k_out)
        self.discrete: DiscreteHead = DiscreteHead(config.d_c, k_classes)

    def forward_layers(self, raw: Constellation) -> list[Constellation]:
        # step 1: the primary path; returns the encoder output plus one entry
        # per layer, length n_layers + 1.
        assert_valid(raw)
        states: list[Constellation] = [self.encoder(raw)]
        current = states[0]
        assert current.x_c is raw.x_c, "the encoder must pass x_c through"
        for layer in self.layers:
            current = layer(current)
            assert current.x_c is raw.x_c, "every layer must keep x_c fixed"
            states.append(current)
        assert len(states) == self.config.n_layers + 1
        return states

    def forward(self, raw: Constellation, debug: bool = False) -> ModelOutput:
        # step 1: run the layers; validate after each layer while debug is True.
        assert_valid(raw)
        current = self.encoder(raw)
        for layer in self.layers:
            current = layer(current, debug=debug)
        # step 2: the heads read the invariant controller state.
        y = self.continuous(current.h_c)
        z = self.discrete(current.h_c)
        assert y.shape == (raw.x.shape[0], self.k_out)
        assert z.shape == (raw.x.shape[0], self.k_classes)
        return ModelOutput(constellation=current, y=y, z=z, preds=predicates(current))


def count_parameters(model: nn.Module) -> int:
    # step 1: count exactly the trainable entries the optimizer sees.
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total > 0, "the model must carry trainable parameters"
    return total


def build_model(config: SpinGnnConfig, k_out: int = 3, k_classes: int = 4) -> SpinGnn:
    # step 1: the only sanctioned constructor; the config seed fixes the draw.
    torch.manual_seed(config.seed)
    return SpinGnn(config, k_out=k_out, k_classes=k_classes)
