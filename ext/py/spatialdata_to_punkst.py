### Convert a Points element in a SpatialData (zarr) store into the plain
### tab/comma-delimited transcript file expected by punkst (pts2tiles,
### convert-dge, etc).
#
# Usage:
#   python spatialdata_to_punkst.py \
#     --sdata /path/to/data.zarr --points-key transcripts \
#     --coordinate-system global --out transcripts.tsv
# where --sdata is the SpatialData zarr store and --points-key names the
# Points element inside it.
#
# The output has columns: x, y, feature[, count], preceded by a single
# "#"-prefixed header line. pts2tiles auto-detects "#"-prefixed lines as
# header/metadata, so no --skip is needed:
#   punkst pts2tiles --in-tsv transcripts.tsv \
#     --icol-x 0 --icol-y 1 --icol-feature 2 \
#     --tile-size 500 --out-prefix transcripts.tiled

import sys
import gzip
import argparse

import pandas as pd


def _open_output(path):
    if path == "-":
        return sys.stdout
    if path.endswith(".gz"):
        return gzip.open(path, "wt")
    return open(path, "w")


def convert(
    sdata_path,
    points_key,
    out_path,
    coordinate_system=None,
    feature_column=None,
    count_column=None,
    z_column=None,
    digits=4,
    min_qv=None,
    qv_column="qv",
):
    import spatialdata as sd
    from spatialdata.transformations import Identity, get_transformation

    sdata = sd.read_zarr(sdata_path)
    if points_key not in sdata.points:
        raise KeyError(
            f"'{points_key}' not found among Points elements: {list(sdata.points)}"
        )
    points = sdata.points[points_key]

    # Resolve which coordinate system to materialize coordinates in. Default
    # to whatever single coordinate system the element is registered under;
    # if there are several (e.g. the element has been aligned into more than
    # one shared coordinate system), the caller must disambiguate. The special
    # value "intrinsic" exports the element's own stored coordinates without
    # applying any transformation (for Xenium these are microns, while the
    # registered "global" system is image pixels).
    transformations = get_transformation(points, get_all=True)
    if coordinate_system is None:
        if len(transformations) != 1:
            raise ValueError(
                "Points element is registered in multiple coordinate systems "
                f"({list(transformations)}); pass --coordinate-system to pick one."
            )
        coordinate_system = next(iter(transformations))
    if coordinate_system in transformations:
        t = transformations[coordinate_system]
        if not isinstance(t, Identity):
            print(
                f"WARNING: exporting in coordinate system '{coordinate_system}', "
                f"which applies {t!r} to the stored coordinates. punkst expects "
                "microns; use --coordinate-system intrinsic to export the stored "
                "coordinates unchanged.",
                file=sys.stderr,
            )
        points = sd.transform(points, to_coordinate_system=coordinate_system)
    elif coordinate_system != "intrinsic":
        raise ValueError(
            f"Unknown coordinate system '{coordinate_system}'; the element is "
            f"registered in {list(transformations)} (or use 'intrinsic')."
        )

    attrs = points.attrs.get("spatialdata_attrs", {})
    if feature_column is None:
        feature_column = attrs.get("feature_key")
    if feature_column is None:
        raise ValueError(
            "Could not infer the feature (gene) column; pass --feature-column."
        )

    columns = ["x", "y"]
    if z_column:
        columns.append(z_column)
    columns.append(feature_column)
    if count_column:
        columns.append(count_column)

    header_names = ["x", "y"] + (["z"] if z_column else []) + ["feature"]
    if count_column:
        header_names.append("count")
    header = "#" + "\t".join(header_names) + "\n"

    fmt = f"%.{digits}f"

    read_columns = columns + ([qv_column] if min_qv is not None else [])

    out = _open_output(out_path)
    try:
        out.write(header)
        for partition in points[read_columns].to_delayed():
            df = partition.compute()
            if min_qv is not None:
                df = df[df[qv_column] >= min_qv][columns]
            if isinstance(df[feature_column].dtype, pd.CategoricalDtype):
                df[feature_column] = df[feature_column].astype(str)
            df.to_csv(out, sep="\t", header=False, index=False, float_format=fmt)
    finally:
        if out is not sys.stdout:
            out.close()


def main(argv):
    parser = argparse.ArgumentParser(prog="spatialdata_to_punkst")
    parser.add_argument("--sdata", required=True, help="Path to the SpatialData .zarr store")
    parser.add_argument("--points-key", required=True, help="Name of the Points element to export")
    parser.add_argument("--out", required=True, help="Output path (.tsv, .tsv.gz, or - for stdout)")
    parser.add_argument(
        "--coordinate-system",
        default=None,
        help="Coordinate system to materialize points in (default: infer if unambiguous); "
        "'intrinsic' exports the stored coordinates without any transformation",
    )
    parser.add_argument(
        "--min-qv",
        type=float,
        default=None,
        help="Drop rows whose quality column is below this value (e.g. 20 for Xenium)",
    )
    parser.add_argument("--qv-column", default="qv", help="Quality column used by --min-qv")
    parser.add_argument(
        "--feature-column",
        default=None,
        help="Column holding the gene/feature name (default: the element's feature_key)",
    )
    parser.add_argument(
        "--count-column",
        default=None,
        help="Optional column holding per-row counts (omit to treat each row as one molecule)",
    )
    parser.add_argument("--z-column", default=None, help="Optional column holding a z coordinate")
    parser.add_argument("--digits", type=int, default=4, help="Decimal precision for x/y/z")
    args = parser.parse_args(argv)

    convert(
        args.sdata,
        args.points_key,
        args.out,
        coordinate_system=args.coordinate_system,
        feature_column=args.feature_column,
        count_column=args.count_column,
        z_column=args.z_column,
        digits=args.digits,
        min_qv=args.min_qv,
        qv_column=args.qv_column,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
