# Published GP objective: implemented interpretation and derivative

Source: user-supplied ASP-DAC 2024 PDF, Sec. II-C / III-A, Eqs. 3–7 and Fig. 2.

The implemented objective is `WA(P; gamma) + B(P; frozen_tree, gamma) + lambda D(P)`.
DREAMPlace provides WA, density, density-weight updates and preconditioning.
`B` sums x/y pair-WA lengths for the raw FLUTE terminal branches of interior
physical pins. The same net weights used by WA multiply the added branches.

For one coordinate displacement `d = p - s`, with frozen Steiner coordinate s,

```
phi(d)  = d * tanh(d / (2 gamma))
t       = d / (2 gamma)
phi'(d) = tanh(t) + t * (1 - tanh(t)^2)
```

This is algebraically Eq. 5, using stable hyperbolic functions instead of
unshifted exponentials. At gamma=1: d=0 gives value/gradient 0; d=1 gives
0.46211715726 / 0.85534102374; d=2 gives 1.52318831191 / 1.18156849757.
A gradient magnitude above 1 is correct, not an error to clamp. The approximation
error is exactly `2 |d| / (1 + exp(|d| / gamma))`, matching Eq. 6.

`dp/d(cell_origin)=1`, so per-pin gradients add on their owning cell. Fixed-node
coordinates are constants in the feasible optimization variables. Fillers have
no pins and receive no contribution from this term; their density gradients stay
in DREAMPlace. FLUTE, terminal ordering, interior classification and Steiner
coordinates are detached for a whole optimizer step. Differentiating through a
rebuild, or rebuilding at finite-difference probe coordinates, would test a
different nonsmooth function and is not the derivative prescribed by the paper.

## Choices the paper leaves underspecified

- **Interior:** the paper does not unambiguously distinguish per-axis from
  whole-pin boundary filtering. We adopt strict whole-pin interior: x and y
  must both lie strictly inside the net bbox. Boundary pins receive WA only.
  Nets with two or three physical pins contribute no correction, as stated
  explicitly after Eq. 7.
- **Virtual Steiner:** in Fig. 2, vs1/vs2 coincide with the adjacent s1/s3.
  There is no published remote tracing or projection algorithm in the six-page
  paper. We use raw adjacent Steiner coordinates.
- **Trunks:** fixed Steiner-to-Steiner terms have zero coordinate gradients.
  We omit those constants. The gradient matches the frozen branch formulation;
  the reported correction scalar is not claimed to equal total smooth StWL.
- **Pin identity:** raw FLUTE first-d rows are internally sorted terminals;
  later rows are Steiner nodes. Equal coordinates do not imply equal identity.
  Preserve zero-length terminal-to-Steiner links and duplicate physical pins.
  A tuple-to-queue mapping recovers original pin identities after quantization.
  This avoids incorrectly moving a trunk when a terminal and Steiner coincide.

Hand-checked FLUTE case, chosen to match Fig. 2's branch pattern:
`[(2,8),(0,5),(4,6),(9,4),(2,2),(6,0)]` selects `(4,6)->(2,6)` and
`(2,2)->(2,4)`. At gamma=1, correction=3.0463766238230594; the two nonzero
pin-gradient components are +1.181568497569791 in x for the former and
-1.181568497569791 in y for the latter.

## Solver integration and attribution

Activate after 200 WA steps by default. Rebuild at configured iteration
boundaries: the paper-only controller defaults to every step, while the combined
route GP runner defaults to every 20 steps and the large experiments explicitly
use 25. A newly assimilated router observation also forces a rebuild before the
next GP step. Periodic refresh freezes Steiner anchors between rebuilds; it is
an explicit experimental choice, not an assertion of the authors' exact schedule.
Publish a frozen gamma/topology pair and refresh both Nesterov secant endpoints
before line-search evaluations. The forward never rebuilds the tree. The real
PlaceObj sums the correction with WA+density before backward and preconditioning.
A runtime first-activation audit compares the actual objective gradient to the
sum of baseline and correction gradients.

Three paired configurations separate effects: standard WA, WA with the same
secant-refresh schedule, and paper GP with that schedule. The runner explicitly
supports one flat Nesterov non-BB stage with static pin geometry. Fence regions,
routability-driven resizing, timing weight updates and multi-stage macro flows
are rejected; their additional mutable geometry/preconditioner state would need
separate integration. This does not alter the mathematical GP model above.

This implements the published GP model under explicit interpretation choices;
it does not claim access to the authors' source or reproduce their undisclosed
experimental settings. The post-GP quadtree refinement is a separate paper
component and is not part of this GP-gradient implementation.
