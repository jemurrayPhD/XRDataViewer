# XRDataViewer

XRDataViewer is a PySide2/pyqtgraph desktop application for exploring and processing multi-dimensional `xarray` datasets. This repository previously bundled a full overview in `docs/index.html`; the section below consolidates the runtime dependencies required to launch the GUI and its optional integrations.

## Dependency summary

### Core runtime
- **PySide2** – mandatory Qt binding used by the application bootstrapper and widgets; the entry point explicitly sets the `PYQTGRAPH_QT_LIB` environment variable to `PySide2` before importing Qt types.
- **pyqtgraph** – plotting layer for all 1D/2D views and the shared plot widgets embedded across the interface.
- **NumPy** – foundational array math for processing steps, ROI calculations, and plot preparation routines.
- **xarray** – primary data model for loaded datasets, interactive bridge utilities, and in-application serialization.

### Optional and feature-specific packages
- **SciPy** – supplies Gaussian, median, Butterworth, and Savitzky–Golay filters; processing functions gracefully degrade when SciPy is unavailable.
- **QtWebEngine** – enables the embedded JupyterLab browser inside the Interactive tab; the application falls back to placeholder messaging if `PySide2.QtWebEngineWidgets` cannot be imported.
- **JupyterLab** – required to launch the integrated notebook server; the interactive manager searches for the `jupyter-lab` command and reports a helpful error when it is missing.
- **OpenCV (cv2)** – optional for exporting sequential view animations to AVI/MP4; the exporter warns users when the module is absent.
- **Zarr** – enables opening `.zarr` archives through xarray’s `open_zarr` helper, which is invoked when datasets with the suffix are selected.
- **IPython** – enhances the bridge utilities by auto-syncing notebook variables into the running GUI when available.

## Feature overview

### Slicing and navigation
- **Sequential view** – step through higher-dimensional datasets by selecting the live axes, fixing the remaining indices with spin boxes, and scrubbing through frames via the timeline slider or keyboard shortcuts.
- **Multi-view grids** – arrange multiple slice panels side by side, each with independent axis selections so orthogonal cuts or synchronized views can be inspected simultaneously.
- **Overlay canvas** – drop images or 1D traces to stack related datasets; drag-to-select multiple layers and adjust visibility or processing from the inspection pane.

### Data processing workflows
- **Dimension-aware catalog** – the processing browser filters available operations based on dataset dimensionality, exposing 1D baseline subtraction, smoothing, and Savitzky–Golay fits alongside multi-dimensional Gaussian, median, and Butterworth filters.
- **Processing stacks** – apply successive operations to the currently selected overlay layers, undo steps individually, clear the stack, or save reusable pipelines that can be replayed on other datasets.
- **Batch application** – select multiple layers (or dataset tree entries) and apply the same processing pipeline in one action when comparing related measurements.

### Plotting and analysis
- **Interactive ROI plots** – create regions of interest on images to generate 1D traces, with automatic axis scaling and marker-aware legends.
- **Line plot previews** – the sequential and overlay panes auto-range new curves, support simultaneous plotting of multiple traces, and respect per-curve styling choices.
- **Annotation tools** – open the annotation dialog from any plot to add text boxes, arrows, and configurable legends while previewing updates live.

### Styling and aesthetics
- **Toolbar groupings** – layout, scaling, and styling controls are grouped by function around each viewer so pan/zoom, color maps, and annotation buttons are easy to discover.
- **Line style dialogs** – choose pens, markers (including marker-only “no line” mode), and opacity per plot, with controls enabled automatically for 1D data across the sequential and overlay panes.
- **Colormap and level editors** – tweak image layers using the color scale widgets and per-layer controls that mirror pyqtgraph’s HistogramLUT functionality.

### Exporting plots and data
- **Static exports** – save figures as images or vector graphics directly from each plot’s context menu or toolbar button.
- **Animation capture** – export sequential view sweeps to video when OpenCV is available, using the configured frame range and playback speed.
- **Processed datasets** – persist the results of processing pipelines by exporting modified xarray DataArrays or saving pipeline definitions for repeatable analysis sessions.

Refer to the HTML documentation in `docs/` for a deeper architectural overview and feature guide.
