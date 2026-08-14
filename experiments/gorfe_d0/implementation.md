# GoRFE-D0-A implementation record

Status: **IMPLEMENTED LOCALLY; SEALED-STATE SERVER RESULT PENDING**

The audit is CPU-only and consumes only the exact GoRFE-V1 `_04` preparation
and evaluation states.  It reconstructs the unchanged V1 nested scores and all
seven selector orders, reports fixed-budget curves and topology proxies, and
requires its 168 original portfolio values per scene to reproduce the sealed V1
JSON within `1e-12`.

The implementation deliberately does not import the renderer, load a dataset,
use CUDA, compute pixel-support overlap, or estimate joint group utility.  Those
missing cross-group quantities are the decision boundary for a possible D0-B,
not values silently approximated by D0-A.
