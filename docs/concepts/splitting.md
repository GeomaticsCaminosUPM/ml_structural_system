# Spatial split

Several features describe a building's neighbors (contact forces, position in the block), so touching buildings share information. A random building-level split would put neighbors on both sides and overestimate the scores.

**Blocks.** Footprints closer than 5 cm are joined into blocks: the connected components of the "touching" graph. A file without a `block_id` column gets one automatically ([`add_block_ids`][ml_structural_system.dataset_prep.blocks.add_block_ids]).

**Split.** With `preprocessing.id_col: block_id`, whole blocks go to one partition. Each source file is split separately, stratified by class (and by `preprocessing.group`, if set):

1. 20% of the labeled buildings (`test_size`) go to the **test** set.
2. Of the rest, 15% of the total (`val_size`) go to the **validation** set.
3. The remaining 65% are the **training** set.

Each block is assigned by the class of its first building. A class or group with a single block goes to training.

**Transfer.** With `preprocessing.data.train_source` and `test_source` instead of `source`, a whole area is held out as the test set. The validation set is then taken from the training area. This is how the paper evaluates transfer to new cities and districts.

```yaml
preprocessing:
  data:
    train_source: [district_a.gpkg]
    test_source: [district_b.gpkg]
    val_size: 0.15
```
