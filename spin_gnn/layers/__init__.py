# the layers package: one message pass and one set of node updates.
# flow:
# 1. message builds the invariant scalar message and the equivariant vector message.
# 2. update applies the gated per-field satellite updates and the controller update.
# 3. controller pools satellite state into the controller.
