# Trench contour continuity (0.1.10)

## Problem and scope

A granular SEM roof edge can contain several local maxima. Per-normal peak
selection can alternate between the rim, its gradient flank and texture. A
binary boundary-mode bonus can overpower path continuity. Broad roof edges
can also have similar gray levels on both sides, so a step-only support test
incorrectly rejects them. These are image interpretation errors; straight
walls are not a universal physical assumption.

The automatic guide workflow now uses image-supported neighborhood consensus.
The legacy manual-ROI tracing path keeps its prior behavior.

## Algorithm

1. Sample image profiles in guide arc-length / normal coordinates. Only
   candidate positions within the displayed constant-radius corridor are used.
   Outside-corridor pixels provide contrast context, never output coordinates.
2. Pool tangential profiles with a median over approximately 9 pixels, then
   Gaussian support (tangent sigma 1.5 px, normal sigma 0.8 px). Work is chunked
   to bound the temporary median allocation. This rejects sparse transverse
   marks and grains without averaging across the whole trench.
3. Accept step contrast or resolved roof prominence. Estimate residual grain
   using second differences, so the slope of a broad real edge does not raise
   its own rejection threshold. Keep cavity connectivity and rim barriers.
4. Rank maximum-gradient candidates continuously by their actual gradient
   strength. Use dynamic programming over up to six candidates per sample,
   with a bounded displacement penalty. A sustained notch may change the
   path; one contaminated sample may be skipped at an explicit cost.
5. Require contrast on each original normal as well. Skipped or unsupported
   samples remain gaps in the trace, CD interpolation and GFE checks. No
   straight-wall fit or drawn-guide coordinate substitution is performed.

Preprocessing parameters and method are saved in each JSON/CSV `guide_traces`
entry. Edge spread remains computed from the original sampled image, not the
pooled profile. The existing project, calibration and display-unit flows apply.

## Validation and limitations

`tests/test_trench_continuity.py` checks noisy straight, tapered, bowed and
notched walls, displaced/reversed guides, blank/noisy unsupported regions and
entrance arcs of 8/15/30/60 px. The existing strict-corridor, shaded/16-bit/
inverted-image, neighboring-trench, full workflow and export tests also apply.

The supplied real SEM remains a partial detection: the left body contour is
more stable, but dimension annotations, the entrance and parts of the floor
can still be ambiguous. These are not silently connected. User images and
their result files are kept locally, outside version control and release data.

Pooling has a spatial bandwidth. This is a shape extraction method, not an
unbiased line-edge-roughness measurement. Fine features below that bandwidth
may be attenuated. Pixel jitter reduction is not a dimensional-accuracy or
confidence-probability claim. Reference-calibrated SEMs are still needed for
physical accuracy validation; printed dimensions are only comparison values.

## Research consulted through SciSpace

This implementation uses general principles from the sources below; it is
not a reproduction of their algorithms or their reported performance.

- Nakamae et al., *Boundary extraction in the SEM cross-section of LSI* (2001),
  [DOI 10.1117/12.444214](https://doi.org/10.1117/12.444214): neighboring-pixel
  evidence and hierarchical noise suppression for SEM roof boundaries.
- Verbin and Zickler, *Field of Junctions: Extracting Boundary Structure at Low
  SNR* (ICCV 2021), [author paper](https://arxiv.org/abs/2011.13866): spatial
  consistency while preserving corners and junctions.
- Mack and Bunday, *Using the analytical linescan model for SEM metrology*
  (2017), [DOI 10.1117/12.2258631](https://doi.org/10.1117/12.2258631): filtering
  changes apparent roughness; calibrated physical models are a separate
  metrology approach. No calibrated physical SEM model is claimed here.
