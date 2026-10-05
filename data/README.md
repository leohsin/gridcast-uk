# Data

This directory contains local datasets used by GridCast UK.

## raw/

Raw annual GB electricity demand data downloaded from the
NESO Data Portal.

These files are not committed to Git and can be reproduced
using:

```bash
uv run python scripts/download_historical_neso.py