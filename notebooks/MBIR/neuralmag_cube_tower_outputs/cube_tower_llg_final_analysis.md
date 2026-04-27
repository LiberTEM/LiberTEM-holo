# Cube-Tower LLG Finalization

## Basis

- Status: finalized_on_fine_llg
- Backend: jax
- Trusted LLG target: notebooks/MBIR/neuralmag_cube_tower_outputs/cube_tower_relaxed_comparison.npz

## Coarse Model Check

- Coarse shape XYZ: [56, 171, 28]
- Coarse cell size (nm): 6.0
- Coarse LLG phase RMS vs imported fine target: 8.88881
- Coarse LLG max abs phase residual vs imported fine target: 14.5201

## Fine LLG Reproduction Check

- Fine shape XYZ: [112, 342, 56]
- Fine cell size (nm): 3.0
- Initialization used: uniform_y
- Fine LLG phase RMS vs saved target: 0.00389179
- Fine LLG max abs phase residual vs saved target: 0.00567245
- Fine saved-target vs LLG mean dot: 0.999602
- Fine saved-target vs LLG vector RMS: 0.000334974
- Coarse/fine phase-RMS ratio: 2283.99

## Conclusion

The coarse-grid LLG model does not reproduce the imported fine-grid target, so that discrepancy is a model/target mismatch. Fine-grid LLG reproduces the saved fine-grid target closely in both phase and magnetization, so the cube analysis should be finalized on the fine-grid LLG result.

Use the fine-grid LLG target stored in cube_tower_relaxed_comparison.npz as the trusted basis for follow-on cube analysis.
