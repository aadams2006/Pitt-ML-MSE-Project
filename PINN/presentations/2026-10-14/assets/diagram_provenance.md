# Introductory PINN training schematic

- Created: October 11, 2026.
- Tool: built-in `image_gen.imagegen` (not CLI fallback).
- Saved project asset: `PINN/presentations/2026-10-14/assets/pinn_training_diagram.png`.
- Purpose: conceptual introduction for engineering researchers without ML experience.
- The node count is illustrative. The architecture slide documents the implemented two 16-unit hidden layers.
- The diagram shows how data and optional physics penalties guide training. The hard output bound is an architectural transform; the figure does not assert that all proposed penalties are active.
- Conceptual background: https://maziarraissi.github.io/PINNs/

## Final generation prompt

Use case: scientific-educational. Create a clean, accurate PINN conceptual diagram image for a University of Pittsburgh engineering research presentation. Wide 2.5:1 aspect ratio, white background. Flat academic diagram, navy #002B5E, muted gold #948151, pale blue and gold fills, Arial-like sans-serif text, generous spacing, large legible labels. Top row flow left to right: box labeled 'Measured inputs' with subtitle 'Film and process conditions', thin navy arrow to a small neural network of connected circles inside a box labeled 'Neural network', then arrow to a box labeled 'Predicted bonded thickness'. Beneath the prediction, two checks side by side labeled 'Agreement with measurements' and 'Agreement with physics'. A branching arrow goes from prediction to both checks. Thin gold feedback arrows from both checks unite and return to the neural network, labeled 'Update network parameters'. This diagram shows iterative training with both data and physics checks. Include only these exact labels. No equations, no PDE claims, no sample numbers, no logo, no title, no gradients or shadows. Keep every arrow direction and label readable. Make a polished scientific schematic that an audience unfamiliar with machine learning can follow.
