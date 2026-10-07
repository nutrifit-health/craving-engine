# Third-party materials

[Русская версия](THIRD_PARTY_NOTICES.ru.md)

The NutriFit restriction covers only rights held by NUTRIFIT LLC. Third-party terms remain independent and take precedence for their material.

## MaleCNS v1.0 anatomical data

File: `data/v9-frozen/connectome.npz`.

Source: [Male CNS Connectome Project](https://male-cns.janelia.org/) and its [download page](https://male-cns.janelia.org/download/). The project page identifies the dataset as CC BY, linking to [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution: FlyEM / HHMI Janelia, University of Cambridge Department of Zoology, MRC Laboratory of Molecular Biology, and Google Research; Berg et al., [Cell, 2026](https://doi.org/10.1016/j.cell.2026.08.015).

Modification: selected ALPN → KC connections into the left mushroom body; original body IDs and synapse counts retained. Other neurons and connections are not included. Normalized weights, synthetic feature adapters and rewired controls are separate computational constructions, not measured physiological weights.

Upstream files:

- `body-annotations-male-cns-v1.0-minconf-0.5.feather`, SHA256 `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2`.
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather`, SHA256 `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1`.

The extracted NPZ records source URLs, selection, license, attribution and limitations in its metadata. Its original SHA256 is `9af289f6854acc05d8656545b1b3bc76aa68de6af6b1cbb280f411f1372c8a9e`; it is also recorded in `data/v9-frozen/manifest.json`.

CC BY 4.0 permits commercial use subject to its terms. NutriFit does not claim exclusive ownership of this data or impose its noncommercial condition on use independently authorized by CC BY 4.0. Retain attribution, indicate changes and provide the license link when redistributing it.

## Runtime dependencies

NumPy, SciPy, Pydantic, scikit-learn, joblib, threadpoolctl and optional PyArrow are installed separately; their source is not vendored here. Their own licenses govern them and are not changed by the NutriFit license. Dependency version constraints are in `pyproject.toml`; the historical environment is in `original-environment.json`.
