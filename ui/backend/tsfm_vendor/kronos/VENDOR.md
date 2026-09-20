# Vendored: Kronos model code

- Source: https://github.com/shiyu-coder/Kronos (MIT, see LICENSE), directory `model/`
- Commit: 67b630e67f6a18c9e9be918d9b4337c960db1e9a (2026-04-13)
- Weights are NOT here: they come from Hugging Face (NeoQuasar/Kronos-*, safetensors only).

Reviewed before vendoring: pure PyTorch/einops/numpy/pandas; no network access, no subprocess,
no eval/exec, no pickle; weights load as safetensors through huggingface_hub's
PyTorchModelHubMixin from a local directory.

Local changes (the only ones):
1. kronos.py: removed `sys.path.append("../")` (mutating the import path of the host process).
2. kronos.py: `from model.module import *` -> `from .module import *` (package-relative).

To update: fetch model/__init__.py, model/kronos.py, model/module.py at a new commit, review the
diff, re-apply the two changes above, and update the commit here.
