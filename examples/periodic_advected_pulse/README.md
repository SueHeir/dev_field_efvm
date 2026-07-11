# Periodic Advected Pulse

![validation comparison](plots/validation.png)

The left panel compares the measured L1 error with the declared first-order acceptance band (including the monotone-refinement check); the right panel compares each measured mass drift with its declared tolerance.

Periodic FIELD wrap gate: fitted order 0.791 (independently recomputed 0.791) against 1.0 +/- 0.25, max mass drift 3.035e-14 <= 1.0e-12; PASS.

## Reference and limits

The transported smooth contact/entropy wave is the analytical Euler reference, as described for linear advection and contact waves in R. J. LeVeque, *Finite Volume Methods for Hyperbolic Problems* (Cambridge, 2002), DOI: [10.1017/CBO9780511791253](https://doi.org/10.1017/CBO9780511791253). This check exercises a serial structured-mesh wrap only; it does not validate AMR, MPI-rank periodic exchange, viscous reconstruction, or corner ghosts. The implementation and validation harness were produced with AI assistance; the exact-solution comparison, independent sweep-side recomputation, and review are the available checks, not a claim of broad solver verification.
