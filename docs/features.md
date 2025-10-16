# XRDataViewer Feature Guide

XRDataViewer provides a desktop-oriented workflow for exploring and processing
multi-dimensional datasets backed by **xarray**.  The application is composed of
several panes that can be shown, hidden, or floated to match the way you prefer
to work.  This guide focuses on the interactive processing tooling that ships
with the application.

## Processing workspace

* **Pipeline sidebar** – Build and tweak ordered processing steps.  Each step
  exposes its parameters through context-aware controls (spin boxes for numeric
  values, combo boxes for enumerations, etc.).  Changes are applied immediately
  to the preview when possible so that you can iterate quickly.
* **ROI preview** – Enable the region of interest (ROI) overlay to compute
  summary statistics across a subset of the data.  Right-click the ROI preview
  plot to change the reduction axis or statistic (mean, median, min, max,
  standard deviation, or peak-to-peak).
* **Colormap switching** – Choose from the bundled scientific colormaps to make
  features stand out.  When a colormap is selected it is applied to the preview
  image instantly.

## Pipeline library

* **Create from scratch** – Use the *Pipeline builder* dialog to chain
  processing functions and capture their parameters as defaults.
* **Edit existing pipelines** – Open the *Pipeline editor* to load a saved
  pipeline, try it on sample data, tweak parameters, and persist the updated
  configuration.
* **Organise by name** – Pipelines are stored under human-readable names and the
  list is kept alphabetically sorted to make it easy to find recipes.

## Data sources

* **Local files** – Load NetCDF, Zarr, and other xarray-compatible datasets
  directly from disk.
* **Bridge integration** – The `xrdataviewer_bridge` helper lets you launch the
  UI straight from Python, handing it in-memory datasets without writing to
  disk first.

## Annotations and export

* **Plot annotations** – Apply textual annotations or crosshair overlays to
  highlight interesting areas before exporting figures.
* **Snapshot saving** – Capture the current view or ROI preview to share the
  result of a processing experiment with colleagues.

For developer-oriented details about how these features are implemented, see the
companion [developer notes](development.md).
