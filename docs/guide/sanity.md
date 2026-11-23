# Checking labels

Field surveys contain mistakes. A building is a **suspect** when most models agree on a class that differs from its survey label: either the label is wrong, or the models share a blind spot. Both are worth a look.

```bash
uv sync --extra sanity        # installs folium, only needed for the map
mlss sanity --config config.yaml --data survey.gpkg
```

It writes `sanity_suspects.csv` and `sanity_check.html`. Open the map in a browser:

- Each suspect is **filled** with the models' consensus color and **outlined** with its survey color. Gray buildings (a layer you can switch on) give context.
- Clicking a suspect shows every model's vote, the agreement and entropy, the height and a Street View link.
- The buttons in the popup record the correct class (or ignore the building). Reviews are saved in the browser, and **Export CSV** downloads them with the building id and coordinates.

From Python, on the output of [`predict`](prediction.md):

```python
from ml_structural_system.sanity import flag_suspects, make_sanity_map

suspects = flag_suspects(buildings, "survey_class", agreement_threshold=0.6)
make_sanity_map(suspects, "survey_class", context=buildings, output_path="sanity_check.html")
```

`survey_class` holds the class name of each surveyed building (see `label_replacements` in the [configuration](configuration.md)). The models are applied to buildings they were trained on too, so checking the whole survey also finds labels the models learned to repeat. Models vote with their tuned decision rule, one vote each.
