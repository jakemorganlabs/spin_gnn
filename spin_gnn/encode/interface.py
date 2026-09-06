# the Encoder Protocol: the hybrid-computing entry point. spec: docs/SPEC.md.
# flow:
# 1. an Encoder maps a raw constellation to a working constellation.
# 2. raw means h, h_c, and v are uninitialized; the encoder seeds them.
# 3. a symbolic encoder places points from predicates; an MLPEncoder would
#    learn the seeding; the IdentityEncoder below is the v0 default.

from typing import Protocol

from spin_gnn.types import Constellation


class Encoder(Protocol):
    # every encoder takes a raw constellation and returns one ready for the layers.
    def forward(self, raw: Constellation) -> Constellation: ...
