from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _fake_package_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _fake_module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class PackageImportSmokeTests(unittest.TestCase):
    def test_top_level_package_import(self):
        sys.path.insert(0, str(SRC_ROOT))
        try:
            import gfm  # noqa: F401
            self.assertTrue(hasattr(gfm, "__file__"))
        finally:
            sys.path.pop(0)

    def test_core_module_import_with_stubs(self):
        sys.path.insert(0, str(SRC_ROOT))
        try:
            # Make sure this import test always executes the module body.
            sys.modules.pop("gfm.gfm", None)

            fake_scanpy = _fake_module("scanpy", AnnData=type("AnnData", (), {}))
            fake_torch = _fake_module("torch")
            fake_tqdm_auto = _fake_module("tqdm.auto", tqdm=lambda x, *args, **kwargs: x)

            fake_torchcfm = _fake_module("torchcfm", OTPlanSampler=object)
            fake_flow_scheduler = _fake_module("flow_matching.path.scheduler", CondOTScheduler=object)
            fake_flow_path = _fake_module("flow_matching.path", AffineProbPath=object)
            fake_flow_solver = _fake_module("flow_matching.solver", ODESolver=object)

            fake_models = _fake_module(
                "gfm.models",
                VAE=object,
                ConditionAwareVAE=object,
                SCVIVAE=object,
                ConditionalODE=object,
                ODEWrapper=object,
                ModelGuidanceWrapper=object,
            )
            fake_helpers = _fake_module(
                "gfm.helpers",
                build_graph=lambda *args, **kwargs: None,
                make_condition_labels_graph=lambda *args, **kwargs: {},
                make_condot_data_loader=lambda *args, **kwargs: None,
                make_data_loader=lambda *args, **kwargs: None,
                make_prediction_data_loader=lambda *args, **kwargs: None,
                get_cell_embedding=lambda *args, **kwargs: None,
                SplitHandler=object,
            )
            fake_vae_training_utils = _fake_module(
                "gfm.vae_training_utils",
                make_vae_dataloader=lambda *args, **kwargs: None,
                make_condition_aware_vae_dataloader=lambda *args, **kwargs: (None, None),
                train_vae=lambda *args, **kwargs: None,
                train_condition_aware_vae=lambda *args, **kwargs: None,
                make_scvi_vae_dataloader=lambda *args, **kwargs: None,
                train_scvi_vae=lambda *args, **kwargs: None,
            )
            fake_train = _fake_module(
                "gfm.train",
                evaluate_metrics_condot=lambda *args, **kwargs: (0.0, 0.0, 0.0),
                evaluate_metrics_no_fm=lambda *args, **kwargs: (0.0, 0.0, 0.0),
                evaluate_one_epoch_condot=lambda *args, **kwargs: 0.0,
                evaluate_one_epoch_no_fm=lambda *args, **kwargs: 0.0,
                train_one_epoch=lambda *args, **kwargs: 0.0,
                evaluate_one_epoch=lambda *args, **kwargs: 0.0,
                evaluate_metrics=lambda *args, **kwargs: (0.0, 0.0, 0.0),
                train_one_epoch_condot=lambda *args, **kwargs: 0.0,
                train_one_epoch_no_fm=lambda *args, **kwargs: 0.0,
            )

            with patch.dict(
                sys.modules,
                {
                    "scanpy": fake_scanpy,
                    "torch": fake_torch,
                    "tqdm.auto": fake_tqdm_auto,
                    "torchcfm": fake_torchcfm,
                    "flow_matching.path.scheduler": fake_flow_scheduler,
                    "flow_matching.path": fake_flow_path,
                    "flow_matching.solver": fake_flow_solver,
                    "gfm.models": fake_models,
                    "gfm.helpers": fake_helpers,
                    "gfm.vae_training_utils": fake_vae_training_utils,
                    "gfm.train": fake_train,
                },
            ):
                module = importlib.import_module("gfm.gfm")

            self.assertTrue(hasattr(module, "GFM"))
        finally:
            sys.path.pop(0)


if __name__ == "__main__":
    unittest.main()