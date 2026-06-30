import pandas as pd
print("hello world")

from fimbox import getAllInputData
getAllInputData(
    boundary="path/to/aoi_boundary.gpkg",
    aoi_id="my_basin",
    out_dir="out/my_basin",
)