# Preparing data

## The footprint file

A GeoPackage (`.gpkg`) or CSV with **one row per building**. Include every footprint of the surveyed area, not only the surveyed ones: the neighbor features of a building depend on the buildings around it, and unsurveyed buildings can be predicted afterwards.

| Column | Needed | Notes |
|---|---|---|
| `geometry` | yes | Footprint polygons, any coordinate reference system |
| label column | for training | Surveyed structural system. Empty for unsurveyed buildings |
| `id` | recommended | Unique building identifier, kept in the outputs |
| features | yes | The columns listed in `preprocessing.feature_columns` |
| `block_id` | no | Group of touching buildings. Computed when missing |
| `dataset` | no | Source name. Defaults to the file name, used to split each file separately |

Several files can be combined (`preprocessing.data.source` accepts a list). They are reprojected to a common coordinate reference system.

## Features

Any numeric, ordinal or categorical column can be a feature. The paper uses 27:

| Group | Columns | Where they come from |
|---|---|---|
| Height | `height` | Your data (meters) |
| Construction era | `year`, `code_quality` | Your data. `code_quality` is derived from `year` when `preprocessing.data.code_type` is set |
| Roof | `roof_material` | Your data |
| Plan shape (18) | `EC8_eccentricityRatio`, `EC8_radiusRatio`, `EC8_compactness`, `ASCE7_setbackRatio`, `ASCE7_holeRatio`, `ASCE7_parallelityAngle`, `GNDTII_beta1_mainShapeSlenderness`, `GNDTII_beta2_setbackRatio`, `GNDTII_beta4_eccentricityRatio`, `GNDTII_beta6_setbackSlenderness`, `CSCR2010_eccentricityRatio`, `NTC23_setbackRatio`, `NTC23_holeRatio`, `polsby_popper`, `convex_hull_irregularity`, `inertia_circle_ratio`, `slenderness_bbox`, `slenderness_inertia` | Computed from the footprint |
| Neighbors (5) | `contact_force`, `contact_confinementRatio`, `contact_angularAcc`, `contact_angle`, `blockPosition` | Computed from the footprints |

With `preprocessing.compute_missing_features: true` (the default), footprint features listed in `feature_columns` but missing from the file are computed with [footprint_attributes](https://github.com/GeomaticsCaminosUPM/footprint_attributes). Height, year and roof material are never estimated: leave them out of `feature_columns` if your data lacks them.

!!! note "Position in the block"
    Recent footprint_attributes versions call the position column `blockPosition`. Older datasets, like the paper's, call it `relativePosition`. Use whichever name your data has in `feature_columns` and `categorical_feature_mappings`.

## Seismic-code level

`code_quality` places each building in the seismic code in force when it was built. Set `preprocessing.data.code_type` to derive it from `year`:

| `code_type` | `low_code` from | `medium_code` from | `high_code` from |
|---|---|---|---|
| `costa_rica` | 1974 | 1986 | 2003 |
| `guatemala` | 1996 | | |
| `dominican_republic` | 1979 | 2011 | |

Earlier buildings are `pre_code`.

## Labels

The classes are yours: nothing in the library fixes them. Survey codes are mapped to the classes the models learn with `preprocessing.label_replacements`, and checked against `preprocessing.label_values`. A replacement to `None` drops the building. For example, the paper groups its codes into four classes (leave `label_values` empty to accept any class):

```yaml
label_values: [ADO, CR, M, S]   # adobe, reinforced concrete, masonry, steel
label_replacements:
  S_light: S      # light steel
  S_frame: CR     # steel frames grouped with reinforced concrete
  MUR: M
  MCF: M
  MR: M
  I: None         # unknown, dropped
```

Classes rarer than `min_class_fraction` of the labeled buildings are dropped, since they cannot be learned or evaluated reliably.
