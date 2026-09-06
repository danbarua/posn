"""Force a headless backend before any test module imports pyplot.

Backend selection is process-wide and import-order dependent: matplotlib
picks a GUI backend (e.g. macosx) by default when a display is present, and
GUI backends report Retina-scaled canvas geometry that does not match saved
raster output pixel-for-pixel. Every plotting test (segmentation, memory,
xor, message) needs the deterministic dpi*inches sizing Agg provides, so this
is set at collection time rather than duplicated per module.
"""

import matplotlib

matplotlib.use("Agg")
