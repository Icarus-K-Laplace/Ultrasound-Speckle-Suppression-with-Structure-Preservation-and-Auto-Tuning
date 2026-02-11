# Design Rationale

## Problem
Ultrasound enhancement has a core trade-off:
- too much denoising -> blurred boundaries
- too much sharpening -> speckle comes back

## Method
1. Log-domain transform for multiplicative speckle.
2. Lee filter as robust local-statistics denoiser.
3. Structure tensor coherence + gradient for edge confidence.
4. Multi-scale detail reinjection with constrained gains.
5. Auto grid search for balanced metrics:
   - CNR non-decreasing
   - ENL increasing
   - Edge ratio close to 1.0

## Why this matters
This framework is practical for research pipelines and easy to migrate to C++/GPU.
