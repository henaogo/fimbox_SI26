
import logging
from pathlib import Path

import fimbox

log = logging.getLogger(__name__)

OUT_DIR = Path(".././out/test_smallB/watershed-data")
bridge_gpkg = OUT_DIR / "osm_bridges_subset.gpkg"
dem_path = OUT_DIR / "dem.tif"
out_dir = OUT_DIR


print(out_dir)