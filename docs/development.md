# XRDataViewer processing internals

This document describes the structure of the processing subsystem so developers
can confidently extend it.  The recent cleanup consolidated previously
duplicated logic and the notes below capture the resulting layout.

## Module overview

* `xrdataviewer/processing.py` – hosts the Qt widgets and dialogs that manage
  processing pipelines, including the sidebar, editor, and builder flows.
* `data_processing.py` – contains the core `ProcessingPipeline` data model and
  helper functions that apply steps to raw numpy arrays.
* `xr_plot_widget.py` – embeds PyQtGraph-based visualisation widgets used by the
  processing dialogs to preview results.

## Parameter forms

* `ParameterForm` is the reusable widget that renders a list of
  `ParameterDefinition` objects into Qt controls.  Each field is represented by a
  `_WidgetBinding` dataclass which bundles the widget instance and value access
  callbacks.  The bindings eliminate the bespoke getter/setter logic that used to
  be duplicated across the class.
* `parametersChanged` is emitted whenever the user edits a value.  The helper
  `_connect_signal` wraps Qt signal connections with a lambda so parameters are
  safely re-emitted regardless of the original signal signature.

## Pipeline management

* `ProcessingManager` maintains an in-memory dictionary of named pipelines and
  emits `pipelines_changed` whenever the registry mutates.  The `list_pipelines`
  and `get_pipeline` helpers always return defensive copies so UI components can
  mutate values without affecting the stored versions until they explicitly call
  `save_pipeline`.
* `PipelineEditorDialog` is responsible for loading data previews, syncing the
  parameter forms back into `ProcessingStep` instances, and rendering ROI
  previews.  Its `_rebuild_forms` helper recreates the UI whenever the pipeline
  definition changes, keeping the widget tree consistent.
* `PipelineBuilderDialog` guides users through creating a new pipeline from
  scratch.  Each available processing function gets its own `ParameterForm`
  instance stored in a stacked widget so the correct form can be shown when the
  user selects a step.

## Extending the system

1. Add a new processing function to `data_processing.py`, providing metadata via
   `ProcessingFunctionSpec` so the UI knows how to render its parameters.
2. If the function requires a novel field type, implement a new helper in
   `ParameterForm._create_binding` that returns a `_WidgetBinding` with suitable
   reader and writer callables.
3. Update the docs in `docs/features.md` to surface the new capability to users
   and mention any workflow adjustments.

With these conventions in mind you can build additional processing features
while keeping the UI code straightforward to reason about.
