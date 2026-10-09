# Upstream provenance

Vendored from https://github.com/sofa-framework/Optimus at commit
`cd41bf17d1ae451aa7208ed0e2219d0702847bbd` (Optimus 1.1, targets SOFA 21.12),
LGPL as in `LICENSE`; original authors in `Authors.txt`.

Only the minimal stochastic-filtering core is kept:

```
src/initOptimusPlugin.{h,cpp}                     entry points (rewritten: registerObjects)
src/optimusConfig.h                               hand-written (was optimusConfig.h.in)
src/genericComponents/FilterEvents.{h,cpp}
src/genericComponents/ObservationSource.h
src/genericComponents/OptimParams.{h,inl,cpp}
src/genericComponents/SimulatedStateObservationSource.{h,inl,cpp}
src/genericComponents/TimeProfiling.h
src/stochasticFiltering/FilteringAnimationLoop.{h,cpp}
src/stochasticFiltering/MappedStateObservationManager.{h,inl,cpp}
src/stochasticFiltering/ObservationManagerBase.h
src/stochasticFiltering/PreStochasticWrapper.{h,cpp}
src/stochasticFiltering/ROUKFilter.{h,inl,cpp}
src/stochasticFiltering/StochasticFilterBase.h
src/stochasticFiltering/StochasticStateWrapperBase.h
src/stochasticFiltering/StochasticStateWrapper.{h,inl,cpp}
```

Dropped: every other filter (UKFClassic, ETKF, ...), Verdandi/Python bindings,
image/optical observation managers, example scenes and tests.

Local modifications beyond the SOFA 25.12 API migration (full list and the
gates that exercise them in `docs/optimus_port.md` §2):

- `StochasticStateWrapper`: `mstate` path data, `getMappedPosFromFilterVector`,
  `PartialFixedProjectiveConstraint` ignored (nodes stay in the state).
- `MappedStateObservationManager`: mapping optional, wrapper-based mapped
  positions, prediction-only when no observation is valid; observations typed by
  the observed state `DataTypes2` (direct-mapping path only for identical
  master/observed types); instantiation `<double, Vec6Types, Vec3Types>` for the
  Vec6 Cosserat strain state.
- `StochasticStateWrapper<Vec6Types, double>` (`stateDim` 6/6) for the Vec6
  strain state.
- `ObservationManager` base: `observationVariances` (per-coordinate R, zero
  weight for unobserved coordinates), `getObservedSize()`.
- `SimulatedStateObservationSource`: `trackedObservationsValid` data.
- `ROUKFilter`: `matUinv.setIdentity()` after both prediction resamplings;
  NIS innovation gate after prediction (`innovationGateSigma`, outputs `nis`,
  `correctionApplied`) in both correction variants.
- Build: `CMakeLists.txt` and `OptimusConfig.cmake.in` rewritten for modular
  SOFA targets; `SOFA_TARGET Optimus` defined so `RequiredPlugin` sees the
  components.

Re-syncing with upstream is not expected; if it is done, diff against the
commit above, not against the 21.12 release archive.
