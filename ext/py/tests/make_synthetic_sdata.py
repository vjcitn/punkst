### Build a small synthetic SpatialData zarr store used to test
### spatialdata_to_punkst.py, without requiring a network download.

import numpy as np
import pandas as pd


def build(out_zarr, n_points=200, scale=2.0, seed=0):
    import spatialdata as sd
    from spatialdata.models import PointsModel
    from spatialdata.transformations import Scale, Identity, set_transformation

    rng = np.random.default_rng(seed)
    genes = np.array(["Gapdh", "Actb", "Malat1", "Blank-1"])
    df = pd.DataFrame(
        {
            "x": rng.uniform(0, 100, n_points),
            "y": rng.uniform(0, 100, n_points),
            "gene": pd.Categorical(rng.choice(genes, n_points)),
        }
    )

    points = PointsModel.parse(df, coordinates={"x": "x", "y": "y"}, feature_key="gene")
    # Register two coordinate systems: the raw "pixels" space the data was
    # authored in, and a "microns" space related by a similarity transform,
    # mimicking how real platforms (e.g. Xenium) store a pixel-to-micron
    # scale factor.
    set_transformation(points, Identity(), "pixels")
    set_transformation(points, Scale([scale, scale], axes=("x", "y")), "microns")

    sdata = sd.SpatialData(points={"transcripts": points})
    sdata.write(out_zarr, overwrite=True)
    return df


if __name__ == "__main__":
    import sys

    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synthetic.zarr")
