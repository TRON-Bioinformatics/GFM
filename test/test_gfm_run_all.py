from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "analysis" / "gfm_run_all.py"
ADATA_PATH = REPO_ROOT / "analysis" / "data" / "preprocessed_replogle_k562_small.h5ad"
SPLIT_DICT_PATH = REPO_ROOT / "analysis" / "data" / "adata_replogle_k562_small_split_dict.pkl"


def load_gfm_run_all_module():
    spec = importlib.util.spec_from_file_location("gfm_run_all_under_test", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    fake_scanpy = types.ModuleType("scanpy")
    fake_scanpy.read_h5ad = lambda *args, **kwargs: None

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, current_device=lambda: 0)
    fake_torch.device = lambda name: name

    fake_pl = types.ModuleType("pytorch_lightning")
    fake_pl.seed_everything = lambda seed: None

    fake_gfm_pkg = types.ModuleType("gfm")
    fake_gfm_pkg.__path__ = []

    fake_gfm_module = types.ModuleType("gfm.gfm")
    fake_gfm_module.GFM = object

    fake_helpers_module = types.ModuleType("gfm.helpers")
    fake_helpers_module.compute_metrics = lambda *args, **kwargs: None

    with patch.dict(
        sys.modules,
        {
            "scanpy": fake_scanpy,
            "torch": fake_torch,
            "pytorch_lightning": fake_pl,
            "gfm": fake_gfm_pkg,
            "gfm.gfm": fake_gfm_module,
            "gfm.helpers": fake_helpers_module,
        },
    ):
        spec.loader.exec_module(module)
    return module


class FakePredictions:
    def __init__(self):
        self.write_paths: list[Path] = []

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake-h5ad", encoding="utf-8")
        self.write_paths.append(path)


class FakeSplit:
    def __init__(self):
        self.split_df = pd.DataFrame(
            {
                "condition": ["train_cond", "test_cond"],
                "split": ["train", "test"],
            }
        )
        self.covariate_columns = ["condition"]
        self.add_split_calls = 0

    def add_split_to_adata(self, adata):
        self.add_split_calls += 1


class FakeGFM:
    last_instance: FakeGFM | None = None

    def __init__(self, adata, **kwargs):
        self.adata = adata
        self.init_kwargs = kwargs
        self.output_dir = kwargs["output_dir"]
        self.model_name = kwargs["model_name"]
        self.split = FakeSplit()
        self.predictions = FakePredictions()
        self.calls: list[tuple[str, dict]] = []
        FakeGFM.last_instance = self

    def initialize_fm(self, **kwargs):
        self.calls.append(("initialize_fm", kwargs))

    def pretrain_vae(self, **kwargs):
        self.calls.append(("pretrain_vae", kwargs))

    def prepare_training(self, **kwargs):
        self.calls.append(("prepare_training", kwargs))

    def train_model(self, **kwargs):
        self.calls.append(("train_model", kwargs))

    def load_model(self):
        self.calls.append(("load_model", {}))

    def predict(self, pred_df):
        self.calls.append(("predict", {"n_rows": len(pred_df)}))
        return self.predictions


class GFMRunAllSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ADATA_PATH.exists():
            raise unittest.SkipTest(f"Missing test dataset: {ADATA_PATH}")
        if not SPLIT_DICT_PATH.exists():
            raise unittest.SkipTest(f"Missing split dict: {SPLIT_DICT_PATH}")

    def setUp(self):
        FakeGFM.last_instance = None
        self.module = load_gfm_run_all_module()

    def test_main_runs_train_predict_metrics_path(self):
        adata = object()
        compute_metrics_calls = []

        def fake_compute_metrics(adata_true, adata_pred, pred_df, **kwargs):
            save_path = Path(kwargs["save_path"])
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text("metric,value\npearson,1.0\n", encoding="utf-8")
            compute_metrics_calls.append(
                {
                    "adata_true": adata_true,
                    "adata_pred": adata_pred,
                    "pred_df_rows": len(pred_df),
                    **kwargs,
                }
            )
            return pd.DataFrame([{"metric": "pearson", "value": 1.0}])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            argv = [
                str(SCRIPT_PATH),
                "--adata-path",
                str(ADATA_PATH),
                "--split-dict-path",
                str(SPLIT_DICT_PATH),
                "--output-dir",
                str(output_dir),
                "--graph-type",
                "one_hot",
                "--save-times",
            ]

            with (
                patch.object(self.module, "GFM", FakeGFM),
                patch.object(self.module, "compute_metrics", side_effect=fake_compute_metrics),
                patch.object(self.module.sc, "read_h5ad", return_value=adata) as read_h5ad_mock,
                patch.object(self.module.pl, "seed_everything") as seed_mock,
                patch.object(sys, "argv", argv),
            ):
                self.module.main()

            seed_mock.assert_called_once_with(42)
            read_h5ad_mock.assert_called_once_with(str(ADATA_PATH), backed="r")

            self.assertIsNotNone(FakeGFM.last_instance)
            instance = cast(FakeGFM, FakeGFM.last_instance)
            self.assertEqual(
                [name for name, _ in instance.calls],
                ["initialize_fm", "pretrain_vae", "prepare_training", "train_model", "predict"],
            )

            pred_path = output_dir / "adata_pred_gfm.h5ad"
            results_path = output_dir / "results_test_gfm.csv"
            timing_path = output_dir / "timing.json"

            self.assertTrue(pred_path.exists())
            self.assertTrue(results_path.exists())
            self.assertTrue(timing_path.exists())
            self.assertEqual(instance.split.add_split_calls, 1)
            self.assertEqual(len(compute_metrics_calls), 1)
            self.assertIs(compute_metrics_calls[0]["adata_true"], adata)
            self.assertIs(compute_metrics_calls[0]["adata_pred"], instance.predictions)
            self.assertEqual(compute_metrics_calls[0]["pred_df_rows"], 1)

            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            self.assertIn("train_time_s", timing)
            self.assertIn("inference_time_s", timing)

    def test_main_runs_reload_predictions_path(self):
        adata = object()
        reloaded_predictions = object()
        compute_metrics_calls = []

        def fake_compute_metrics(adata_true, adata_pred, pred_df, **kwargs):
            compute_metrics_calls.append(
                {
                    "adata_true": adata_true,
                    "adata_pred": adata_pred,
                    "pred_df_rows": len(pred_df),
                    **kwargs,
                }
            )
            return pd.DataFrame([{"metric": "pearson", "value": 1.0}])

        def fake_read_h5ad(path, backed=None):
            if path == str(ADATA_PATH):
                self.assertEqual(backed, "r")
                return adata
            self.assertEqual(backed, None)
            self.assertTrue(str(path).endswith("adata_pred_gfm.h5ad"))
            return reloaded_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            pred_path = output_dir / "adata_pred_gfm.h5ad"
            pred_path.write_text("existing-fake-h5ad", encoding="utf-8")
            argv = [
                str(SCRIPT_PATH),
                "--adata-path",
                str(ADATA_PATH),
                "--split-dict-path",
                str(SPLIT_DICT_PATH),
                "--output-dir",
                str(output_dir),
                "--graph-type",
                "one_hot",
                "--skip-vae-pretrain",
                "--skip-model-training",
                "--skip-prediction",
                "--save-times",
            ]

            with (
                patch.object(self.module, "GFM", FakeGFM),
                patch.object(self.module, "compute_metrics", side_effect=fake_compute_metrics),
                patch.object(self.module.sc, "read_h5ad", side_effect=fake_read_h5ad),
                patch.object(self.module.pl, "seed_everything"),
                patch.object(sys, "argv", argv),
            ):
                self.module.main()

            self.assertIsNotNone(FakeGFM.last_instance)
            instance = cast(FakeGFM, FakeGFM.last_instance)
            self.assertEqual(
                [name for name, _ in instance.calls],
                ["initialize_fm", "load_model"],
            )
            self.assertEqual(instance.split.add_split_calls, 1)
            self.assertEqual(len(compute_metrics_calls), 1)
            self.assertIs(compute_metrics_calls[0]["adata_true"], adata)
            self.assertIs(compute_metrics_calls[0]["adata_pred"], reloaded_predictions)
            self.assertEqual(compute_metrics_calls[0]["pred_df_rows"], 1)


if __name__ == "__main__":
    unittest.main()
