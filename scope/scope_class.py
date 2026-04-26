#!/usr/bin/env python
from contextlib import contextmanager
import datetime
import numpy as np
import os
import pandas as pd
import pathlib

# Re-export for backward compatibility of imports from this module.
from penquins import Kowalski  # noqa: F401
from scope.surveys.kowalski import make_kowalski_client
import subprocess
import sys
import tdtax
from typing import Optional, Sequence, Union
from .utils import (
    forgiving_true,
    read_hdf,
    read_parquet,
    write_parquet,
    parse_load_config,
)
from .surveys.fritz import radec_to_iau_name
from .surveys.rubin import make_rubin_client, HAS_PYVO
import json
import shutil
import argparse


@contextmanager
def status(message):
    """
    Borrowed from https://github.com/cesium-ml/baselayer/

    :param message: message to print
    :return:
    """
    print(f"[·] {message}", end="")
    sys.stdout.flush()
    try:
        yield
    except Exception:
        print(f"\r[✗] {message}")
        raise
    else:
        print(f"\r[✓] {message}")


class Scope:
    def __init__(self):
        # load configuration
        with status("Loading configuration"):
            self.base_path = pathlib.Path.cwd()
            self.config = parse_load_config()

            self.default_path_dataset = (
                self.base_path / self.config["training"]["dataset"]
            )

            # Rubin TAP token from environment variable
            rubin_token_env = os.environ.get("RUBIN_TAP_TOKEN")
            if rubin_token_env is not None:
                self.config["rubin"]["token"] = rubin_token_env

        # Try setting up Kowalski connection (applies *_INSTANCE_TOKEN env vars internally)
        with status("Setting up Kowalski connection"):
            self.kowalski = make_kowalski_client(self.config)
        if self.kowalski is None:
            # raise ConnectionError("Could not connect to Kowalski.")
            print("Kowalski not available")

        # Try setting up Rubin client (local parquet or TAP)
        rubin_config = self.config.get("rubin", {})
        rubin_data_path = rubin_config.get("data_path") or os.environ.get(
            "RUBIN_DATA_PATH"
        )
        if rubin_data_path:
            with status("Setting up Rubin local client"):
                self.rubin = make_rubin_client(config=rubin_config)
        elif rubin_config.get("token") is not None and HAS_PYVO:
            with status("Setting up Rubin TAP connection"):
                self.rubin = make_rubin_client(config=rubin_config)
        else:
            self.rubin = None
            if rubin_config.get("token") is not None and not HAS_PYVO:
                print("Rubin TAP not available (pyvo not installed)")

    def _get_features(
        self,
        positions: Sequence[Sequence[float]],
        catalog: str = None,
        max_distance: Union[float, int] = 5.0,
        distance_units: str = "arcsec",
        period_suffix: str = None,
    ) -> pd.DataFrame:
        """Get nearest source in feature set for a set of given positions

        :param positions: R.A./Decl. [deg]
        :param catalog: feature catalog to query
        :param max_distance:
        :param distance_units: arcsec | arcmin | deg | rad
        :return:
        """
        if self.kowalski is None:
            raise ConnectionError("Kowalski connection not established.")
        if catalog is None:
            catalog = self.config["kowalski"]["collections"]["features"]

        period_colname = "period"
        if not ((period_suffix is None) | (period_suffix == "None")):
            period_colname = f"{period_colname}_{period_suffix}"

        features_dct = {}
        query = {
            "query_type": "near",
            "query": {
                "max_distance": max_distance,
                "distance_units": distance_units,
                "radec": positions,
                "catalogs": {
                    catalog: {
                        "filter": {},
                        "projection": {
                            period_colname: 1,
                            "ra": 1,
                            "dec": 1,
                        },
                    }
                },
            },
        }
        responses = self.kowalski.query(query=query)
        for name in responses.keys():
            if len(responses[name]) > 0:
                response = responses[name]
                if response.get("status", "error") == "success":
                    features_response = response.get("data").get(catalog)
                    features_dct.update(features_response)
        features_nearest = [v[0] for k, v in features_response.items() if len(v) > 0]
        df = pd.DataFrame.from_records(features_nearest)

        return df

    def _get_nearest_gaia(
        self,
        positions: Sequence[Sequence[float]],
        catalog: str = None,
        max_distance: Union[float, int] = 5.0,
        distance_units: str = "arcsec",
    ) -> pd.DataFrame:
        """Get nearest Gaia source for a set of given positions

        :param positions: R.A./Decl. [deg]
        :param catalog: Gaia catalog to query
        :param max_distance:
        :param distance_units: arcsec | arcmin | deg | rad
        :return:
        """
        if self.kowalski is None:
            raise ConnectionError("Kowalski connection not established.")
        if catalog is None:
            catalog = self.config["kowalski"]["collections"]["gaia"]

        gaia_dct = {}
        query = {
            "query_type": "near",
            "query": {
                "max_distance": max_distance,
                "distance_units": distance_units,
                "radec": positions,
                "catalogs": {
                    catalog: {
                        "filter": {},
                        "projection": {
                            "parallax": 1,
                            "parallax_error": 1,
                            "pmra": 1,
                            "pmra_error": 1,
                            "pmdec": 1,
                            "pmdec_error": 1,
                            "phot_g_mean_mag": 1,
                            "phot_bp_mean_mag": 1,
                            "phot_rp_mean_mag": 1,
                            "ra": 1,
                            "dec": 1,
                        },
                    }
                },
            },
            "kwargs": {"limit": 1},
        }
        responses = self.kowalski.query(query=query)
        for name in responses.keys():
            if len(responses[name]) > 0:
                response = responses[name]
                if response.get("status", "error") == "success":
                    gaia_response = response.get("data").get(catalog)
                    gaia_dct.update(gaia_response)
        gaia_nearest = [v[0] for k, v in gaia_dct.items() if len(v) > 0]
        df = pd.DataFrame.from_records(gaia_nearest)

        df["M"] = df["phot_g_mean_mag"] + 5 * np.log10(df["parallax"] * 0.001) + 5
        df["Ml"] = (
            df["phot_g_mean_mag"]
            + 5 * np.log10((df["parallax"] + df["parallax_error"]) * 0.001)
            + 5
        )
        df["BP-RP"] = df["phot_bp_mean_mag"] - df["phot_rp_mean_mag"]

        return df

    def _get_light_curve_data(
        self,
        ra: float,
        dec: float,
        catalog: str = None,
        cone_search_radius: Union[float, int] = 2,
        cone_search_unit: str = "arcsec",
        filter_flagged_data: bool = True,
    ) -> pd.DataFrame:
        """Get light curve data from Kowalski

        :param ra: R.A. in deg
        :param dec: Decl. in deg
        :param catalog: collection name on Kowalski
        :param cone_search_radius:
        :param cone_search_unit: arcsec | arcmin | deg | rad
        :param filter_flagged_data: remove flagged/bad data?
        :return: flattened light curve data as pd.DataFrame
        """
        if self.kowalski is None:
            raise ConnectionError("Kowalski connection not established.")
        if catalog is None:
            catalog = self.config["kowalski"]["collections"]["sources"]

        light_curves_raw = []
        query = {
            "query_type": "cone_search",
            "query": {
                "object_coordinates": {
                    "cone_search_radius": cone_search_radius,
                    "cone_search_unit": cone_search_unit,
                    "radec": {"target": [ra, dec]},
                },
                "catalogs": {
                    catalog: {
                        "filter": {},
                        "projection": {
                            "_id": 1,
                            "filter": 1,
                            "field": 1,
                            "data.hjd": 1,
                            "data.fid": 1,
                            "data.mag": 1,
                            "data.magerr": 1,
                            "data.ra": 1,
                            "data.dec": 1,
                            "data.programid": 1,
                            "data.catflags": 1,
                        },
                    }
                },
            },
        }
        responses = self.kowalski.query(query=query)

        for name in responses.keys():
            if len(responses[name]) > 0:
                response = responses[name]
                if response.get("status", "error") == "success":
                    lcs = response.get("data").get(catalog).get("target")
                    light_curves_raw += lcs

        light_curves = []
        for light_curve in light_curves_raw:
            df = pd.DataFrame.from_records(light_curve["data"])
            # broadcast to all data points:
            df["_id"] = light_curve["_id"]
            df["filter"] = light_curve["filter"]
            df["field"] = light_curve["field"]
            light_curves.append(df)

        df = pd.concat(light_curves, ignore_index=True)

        if filter_flagged_data:
            mask_flagged_data = df["catflags"] != 0
            df = df.loc[~mask_flagged_data]

        return df

    def _drop_low_probs(self, ser, class_list, threshold):
        if ser.name in class_list:
            new_values = [0.0 if v < threshold else v for v in ser.values]
            return new_values
        else:
            return ser

    @staticmethod
    def develop():
        """Install developer tools"""
        subprocess.run(["pre-commit", "install"])

    @classmethod
    def lint(cls):
        """Lint sources"""
        try:
            import pre_commit  # noqa: F401
        except ImportError:
            cls.develop()

        try:
            subprocess.run(["pre-commit", "run", "--all-files"], check=True)
        except subprocess.CalledProcessError:
            sys.exit(1)

    def doc(self):
        """Build docs"""

        from .utils import (
            make_tdtax_taxonomy,
            plot_gaia_density,
            plot_gaia_hr,
            plot_light_curve_data,
            plot_periods,
        )

        period_suffix_config = self.config["features"]["info"]["period_suffix"]

        # generate taxonomy.html
        with status("Generating taxonomy visualization"):
            path_assets = self.base_path / "docs" / "assets"
            if not path_assets.exists():
                path_assets.mkdir(parents=True, exist_ok=True)
            tdtax.write_viz(
                make_tdtax_taxonomy(self.config["taxonomy"]),
                outname=path_assets / "taxonomy.html",
            )

        # generate images for the Field Guide
        if self.kowalski is None:
            print("Kowalski not available, skipping image generation.")
            print("Building docs from existing content...")
            subprocess.run(["mkdocs", "build"], check=True)
            return

        period_limits = {
            "cepheid": [1.0, 100.0],
            "delta_scuti": [0.03, 0.3],
            "beta_lyr": [0.3, 25],
            "rr_lyr": [0.2, 1.0],
            "w_uma": [0.2, 0.8],
        }
        period_loglimits = {
            "cepheid": True,
            "delta_scuti": False,
            "beta_lyr": True,
            "rr_lyr": False,
            "w_uma": False,
        }

        # example periods
        with status("Generating example period histograms"):
            path_doc_data = self.base_path / "docs" / "images"

            # stored as ra/decs in csv format under /data/golden
            golden_sets = self.base_path / "data" / "golden"
            for golden_set in golden_sets.glob("*.csv"):
                golden_set_name = golden_set.stem
                display_name = golden_set_name.replace("_", " ").title()
                positions = pd.read_csv(golden_set).to_numpy().tolist()
                features = self._get_features(
                    positions=positions, period_suffix=period_suffix_config
                )

                if len(features) == 0:
                    print(f"No features for {golden_set_name}")
                    continue

                limits = period_limits.get(golden_set_name)
                loglimits = period_loglimits.get(golden_set_name)

                plot_periods(
                    features=features,
                    limits=limits,
                    loglimits=loglimits,
                    title=f"Period Distribution: {display_name}",
                    save=path_doc_data / f"period__{golden_set_name}",
                    period_suffix=period_suffix_config,
                )

        # example skymaps for all Golden sets
        with status("Generating skymaps diagrams for Golden sets"):
            path_doc_data = self.base_path / "docs" / "images"

            path_gaia_density = self.base_path / "data" / "Gaia_hp8_densitymap.fits"
            # stored as ra/decs in csv format under /data/golden
            golden_sets = self.base_path / "data" / "golden"
            for golden_set in golden_sets.glob("*.csv"):
                golden_set_name = golden_set.stem
                display_name = golden_set_name.replace("_", " ").title()
                positions = pd.read_csv(golden_set).to_numpy().tolist()

                plot_gaia_density(
                    positions=positions,
                    path_gaia_density=path_gaia_density,
                    title=f"Sky Distribution: {display_name}",
                    save=path_doc_data / f"radec__{golden_set_name}",
                )

        # example light curves
        with status("Generating example light curves"):
            path_doc_data = self.base_path / "docs" / "images"

            for sample_object_name, sample_object in self.config["docs"][
                "field_guide"
            ].items():
                sample_light_curves = self._get_light_curve_data(
                    ra=sample_object["coordinates"][0],
                    dec=sample_object["coordinates"][1],
                    # catalog=self.config["kowalski"]["collections"]["sources"],
                )
                plot_light_curve_data(
                    light_curve_data=sample_light_curves,
                    period=sample_object.get("period"),
                    title=sample_object.get("title"),
                    save=path_doc_data / sample_object_name,
                )

        # example HR diagrams for all Golden sets
        with status("Generating HR diagrams for Golden sets"):
            path_gaia_hr_histogram = (
                self.base_path / "docs" / "images" / "gaia_hr_histogram.dat"
            )
            # stored as ra/decs in csv format under /data/golden
            golden_sets = self.base_path / "data" / "golden"
            for golden_set in golden_sets.glob("*.csv"):
                golden_set_name = golden_set.stem
                display_name = golden_set_name.replace("_", " ").title()
                positions = pd.read_csv(golden_set).to_numpy().tolist()
                gaia_sources = self._get_nearest_gaia(positions=positions)

                plot_gaia_hr(
                    gaia_data=gaia_sources,
                    path_gaia_hr_histogram=path_gaia_hr_histogram,
                    title=f"HR Diagram: {display_name}",
                    save=path_doc_data / f"hr__{golden_set_name}",
                )

        # build docs
        subprocess.run(["mkdocs", "build"], check=True)

    @staticmethod
    def fetch_models(gcs_path: str = "gs://ztf-scope/models"):
        """
        (deprecated) Fetch SCoPe models from GCP

        :return:
        """
        path_models = pathlib.Path.cwd() / "models"
        if not path_models.exists():
            path_models.mkdir(parents=True, exist_ok=True)

        command = [
            "gsutil",
            "-m",
            "cp",
            "-n",
            "-r",
            os.path.join(gcs_path, "*.csv"),
            str(path_models),
        ]
        p = subprocess.run(command, check=True)
        if p.returncode != 0:
            raise RuntimeError("Failed to fetch SCoPe models")

    @staticmethod
    def fetch_datasets(gcs_path: str = "gs://ztf-scope/datasets"):
        """
        (deprecated) Fetch SCoPe datasets from GCP

        :return:
        """
        path_datasets = pathlib.Path.cwd() / "data" / "training"
        if not path_datasets.exists():
            path_datasets.mkdir(parents=True, exist_ok=True)

        command = [
            "gsutil",
            "-m",
            "cp",
            "-n",
            "-r",
            os.path.join(gcs_path, "*.csv"),
            str(path_datasets),
        ]
        p = subprocess.run(command, check=True)
        if p.returncode != 0:
            raise RuntimeError("Failed to fetch SCoPe datasets")

    def parse_run_train(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--tag",
            type=str,
            help="classifier designation, refers to 'class' in config.taxonomy",
        )
        parser.add_argument(
            "--path-dataset",
            type=str,
            help="local path to .parquet, .h5 or .csv file with the dataset",
        )
        parser.add_argument(
            "--algorithm",
            type=str,
            default="dnn",
            help="name of ML algorithm to use",
        )
        parser.add_argument(
            "--gpu",
            type=int,
            help="GPU id to use, zero-based. check tf.config.list_physical_devices('GPU') for available devices",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="if set, print verbose output",
        )
        parser.add_argument(
            "--job-type",
            type=str,
            default="train",
            help="name of job type for WandB",
        )
        parser.add_argument(
            "--group",
            type=str,
            default="experiment",
            help="name of group for WandB",
        )
        parser.add_argument(
            "--run-sweeps",
            action="store_true",
            help="if set, run WandB sweeps instead of training",
        )
        parser.add_argument(
            "--period-suffix",
            type=str,
            help="suffix of period/Fourier features to use for training",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            help="classification threshold separating positive from negative examples",
        )
        parser.add_argument(
            "--balance",
            type=float,
            default=-1,
            help="factor by which to weight majority vs. minority examples",
        )
        parser.add_argument(
            "--weight-per-class",
            action="store_true",
            help="if set, weight training data based on fraction of positive/negative samples",
        )
        parser.add_argument(
            "--scale-features",
            type=str,
            help="method by which to scale input features (min_max or median_std)",
        )
        parser.add_argument(
            "--test-size",
            type=float,
            help="fractional size of test set, taken from initial learning set",
        )
        parser.add_argument(
            "--val-size",
            type=float,
            help="fractional size of val set, taken from initial learning set less test set",
        )
        parser.add_argument(
            "--random-state",
            type=int,
            help="random seed to set for reproducibility",
        )
        parser.add_argument(
            "--feature-stats",
            type=str,
            help="feature stats to use to standardize features. If set to 'config', source feature stats from values in config file. Otherwise, compute them from data, taking balance into account",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            help="batch size to use for training",
        )
        parser.add_argument(
            "--shuffle-buffer-size",
            type=int,
            help="buffer size to use when shuffling training set",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            help="number of training epochs",
        )
        parser.add_argument(
            "--float-convert-types",
            type=int,
            nargs=2,
            help="convert floats from a to b bits (e.g. 64 32)",
        )
        parser.add_argument(
            "--lr",
            type=float,
            help="dnn learning rate",
        )
        parser.add_argument(
            "--beta-1",
            type=float,
            help="dnn beta_1",
        )
        parser.add_argument(
            "--beta-2",
            type=float,
            help="dnn beta_2",
        )
        parser.add_argument(
            "--epsilon",
            type=float,
            help="dnn epsilon",
        )
        parser.add_argument(
            "--decay",
            type=float,
            help="dnn decay",
        )
        parser.add_argument(
            "--momentum",
            type=float,
            help="dnn momentum",
        )
        parser.add_argument(
            "--monitor",
            type=float,
            help="dnn monitor quantity",
        )
        parser.add_argument(
            "--patience",
            type=int,
            help="dnn patience (in epochs)",
        )
        parser.add_argument(
            "--callbacks",
            type=str,
            nargs="+",
            help="dnn callbacks",
        )
        parser.add_argument(
            "--run-eagerly",
            action="store_true",
            help="dnn run_eagerly",
        )
        parser.add_argument(
            "--pre-trained-model",
            type=str,
            help="name of dnn pre-trained model to load, if any",
        )
        parser.add_argument(
            "--save",
            action="store_true",
            help="if set, save trained model",
        )
        parser.add_argument(
            "--plot",
            action="store_true",
            help="if set, generate/save diagnostic training plots",
        )
        parser.add_argument(
            "--plot-model",
            action="store_true",
            help="if set, plot model architecture",
        )
        parser.add_argument(
            "--weights-only",
            action="store_true",
            help="if set and pre-trained model specified, load only weights",
        )
        parser.add_argument(
            "--skip-cv",
            action="store_true",
            help="if set, skip XGB cross-validation",
        )

        args, _ = parser.parse_known_args()
        self.train(**vars(args))

    def train(
        self,
        tag: str,
        path_dataset: Union[str, pathlib.Path] = None,
        algorithm: str = "dnn",
        gpu: Optional[int] = None,
        verbose: bool = False,
        job_type: str = "train",
        group: str = "experiment",
        run_sweeps: bool = False,
        period_suffix: str = None,
        threshold: float = 0.7,
        balance: Union[float, str] = -1,
        weight_per_class=False,
        scale_features: str = "min_max",
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42,
        feature_stats: str = None,
        batch_size: int = 64,
        shuffle_buffer_size: int = 512,
        epochs: int = 100,
        float_convert_types: list = [64, 32],
        lr: float = 3e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        decay: float = 0.0,
        amsgrad: float = 3e-4,
        momentum: float = 0.9,
        monitor: str = "val_loss",
        patience: int = 20,
        callbacks: list = ["reduce_lr_on_plateau", "early_stopping"],
        run_eagerly: bool = False,
        pre_trained_model: str = None,
        save: bool = False,
        plot: bool = False,
        plot_model: bool = False,
        weights_only: bool = False,
        skip_cv: bool = False,
        evaluate_test: bool = True,
        **kwargs,
    ):
        """Train classifier

        :param tag: classifier designation, refers to "class" in config.taxonomy (str)
        :param path_dataset: local path to .parquet, .h5 or .csv file with the dataset (str)
        :param algorithm: name of ML algorithm to use (str)
        :param gpu: GPU id to use, zero-based. check tf.config.list_physical_devices('GPU') for available devices (int)
        :param verbose: if set, print verbose output (bool)
        :param job_type: name of job type for WandB (str)
        :param group: name of group for WandB (str)
        :param run_sweeps: if set, run WandB sweeps instead of training (bool)
        :param period_suffix: suffix of period/Fourier features to use for training (str)
        :param threshold: classification threshold separating positive from negative examples (float)
        :param balance: factor by which to weight majority vs. minority examples (float or None)
        :param weight_per_class: if set, weight training data based on fraction of positive/negative samples (bool)
        :param scale_features: method by which to scale input features [min_max or median_std] (str)
        :param test_size: fractional size of test set, taken from initial learning set (float)
        :param val_size: fractional size of val set, taken from learning set less test set (float)
        :param random_state: random seed to set for reproducibility (int)
        :param feature_stats: feature stats to use to standardize features. If set to 'config', source feature stats from values in config file. Otherwise, compute them from data, taking balance into account (str)
        :param batch_size: batch size to use for training (int)
        :param shuffle_buffer_size: buffer size to use when shuffling training set (int)
        :param epochs: number of training epochs (int)
        :param float_convert_types: convert from a-bit to b-bit [e.g. 64 to 32] (list)
        :param lr: dnn learning rate (float)
        :param beta_1: dnn beta_1 (float)
        :param beta_2: dnn beta_2 (float)
        :param epsilon: dnn epsilon (float)
        :param decay: dnn decay (float)
        :param amsgrad: dnn amsgrad (float)
        :param momentum: dnn momentum (float)
        :param monitor: dnn monitor quantity (str)
        :param patience: dnn patience [in epochs] (int)
        :param callbacks: dnn callbacks (list)
        :param run_eagerly: dnn run_eagerly (bool)
        :param pre_trained_model: name of dnn pre-trained model to load, if any (str)
        :param save: if set, save trained model (bool)
        :param plot: if set, generate/save diagnostic training plots (bool)
        :param plot_model: if set, plot model architecture (bool)
        :param weights_only: if set and pre-trained model specified, load only weights (bool)
        :param skip_cv: if set, skip XGB cross-validation (bool)
        :param evaluate_test: if True (default), log test/dropped-sample metrics to WandB.
            Set to False to skip WandB logging (e.g. during unit tests). (bool)

        :return:
        """

        from .xgb import XGB
        from .utils import Dataset

        # backward compatibility: accept test=True to mean evaluate_test=False
        if "test" in kwargs:
            evaluate_test = not kwargs.pop("test")

        if path_dataset is None:
            path_dataset = self.default_path_dataset

        config_params = self.config["training"]["classes"][tag]
        train_config_dnn = self.config["training"]["dnn"]
        train_config_xgb = self.config["training"]["xgboost"]

        if period_suffix is None:
            period_suffix = self.config["features"]["info"]["period_suffix"]

        if algorithm in ["DNN", "NN", "dnn", "nn"]:
            algorithm = "dnn"
        elif algorithm in ["XGB", "xgb", "XGBoost", "xgboost", "XGBOOST"]:
            algorithm = "xgb"
        else:
            raise ValueError("Current supported algorithms are DNN and XGB.")

        all_features = self.config["features"][config_params["features"]]
        features = [
            key for key in all_features if forgiving_true(all_features[key]["include"])
        ]
        if not ((period_suffix is None) | (period_suffix == "None")):
            periodic_bool = [all_features[x]["periodic"] for x in features]
            for j, name in enumerate(features):
                if periodic_bool[j]:
                    features[j] = f"{name}_{period_suffix}"

        ds = Dataset(
            tag=tag,
            path_dataset=path_dataset,
            features=features,
            verbose=verbose,
            algorithm=algorithm,
            period_suffix=period_suffix,
        )

        label = config_params["label"]

        # values from argparse args override those defined in config. if latter is absent, use reasonable default
        if threshold is None:
            threshold = config_params.get("threshold", 0.7)
        if balance == -1:
            balance = config_params.get("balance", None)
        if not weight_per_class:
            weight_per_class = config_params.get("weight_per_class", False)
        if scale_features is None:
            scale_features = config_params.get("scale_features", "min_max")
        if test_size is None:
            test_size = config_params.get("test_size", 0.1)
        if val_size is None:
            val_size = config_params.get("val_size", 0.1)
        if random_state is None:
            random_state = config_params.get("random_state", 42)
        if feature_stats == "config":
            feature_stats = self.config.get("feature_stats", None)
        if batch_size is None:
            batch_size = config_params.get("batch_size", 64)
        if shuffle_buffer_size is None:
            shuffle_buffer_size = config_params.get("shuffle_buffer_size", 512)
        if epochs is None:
            epochs = config_params.get("epochs", 100)
        if float_convert_types is None:
            float_convert_types = config_params.get("float_convert_types", [64, 32])

        datasets, indexes, steps_per_epoch, class_weight = ds.make(
            target_label=label,
            threshold=threshold,
            balance=balance,
            weight_per_class=weight_per_class,
            scale_features=scale_features,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            feature_stats=feature_stats,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            epochs=epochs,
            float_convert_types=float_convert_types,
            algorithm=algorithm,
        )

        if lr is None:
            lr = float(config_params.get("lr", 3e-4))
        if beta_1 is None:
            beta_1 = float(config_params.get("beta_1", 0.9))
        if beta_2 is None:
            beta_2 = float(config_params.get("beta_2", 0.999))
        if epsilon is None:
            epsilon = float(config_params.get("epsilon", 1e-7))
        if decay is None:
            decay = float(config_params.get("decay", 0.0))
        if amsgrad is None:
            amsgrad = float(config_params.get("amsgrad", 3e-4))
        if momentum is None:
            momentum = float(config_params.get("momentum", 0.9))
        if monitor is None:
            monitor = config_params.get("monitor", "val_loss")
        if patience is None:
            patience = int(config_params.get("patience", 20))
        if callbacks is None:
            callbacks = tuple(
                config_params.get(
                    "callbacks", ["reduce_lr_on_plateau", "early_stopping"]
                )
            )
        else:
            callbacks = tuple(callbacks)
        if not run_eagerly:
            run_eagerly = config_params.get("run_eagerly", False)
        if pre_trained_model is None:
            pre_trained_model = config_params.get("pre_trained_model")
        if not save:
            save = config_params.get("save", False)
        if not plot:
            plot = config_params.get("plot", False)
        if not weights_only:
            weights_only = config_params.get("weights_only", False)

        # Define default parameters for all DNN models
        dense_branch = train_config_dnn.get("dense_branch", True)
        conv_branch = train_config_dnn.get("conv_branch", True)
        loss = train_config_dnn.get("loss", "binary_crossentropy")
        optimizer = train_config_dnn.get("optimizer", "adam")

        # xgb-specific arguments (descriptions adapted from https://xgboost.readthedocs.io/en/stable/parameter.html and https://xgboost.readthedocs.io/en/stable/python/python_api.html)
        # max_depth: maximum depth of a tree
        max_depth_config = train_config_xgb["gridsearch_params_start_stop_step"].get(
            "max_depth", [3, 8, 2]
        )
        max_depth_start = max_depth_config[0]
        max_depth_stop = max_depth_config[1]
        max_depth_step = max_depth_config[2]

        # min_child_weight: minimum sum of instance weight (hessian) needed in a child
        min_child_weight_config = train_config_xgb[
            "gridsearch_params_start_stop_step"
        ].get("min_child_weight", [1, 6, 2])
        min_child_weight_start = min_child_weight_config[0]
        min_child_weight_stop = min_child_weight_config[1]
        min_child_weight_step = min_child_weight_config[2]

        # eta = kwargs.get("xgb_eta", 0.1)
        eta_list = train_config_xgb["other_training_params"].get(
            "eta_list", [0.3, 0.2, 0.1, 0.05]
        )

        # subsample: Subsample ratio of the training instances (setting to 0.5 means XGBoost would randomly sample half of the training data prior to growing trees)
        subsample_config = train_config_xgb["gridsearch_params_start_stop_step"].get(
            "subsample", [6, 11, 2]
        )
        subsample_start = subsample_config[0]
        subsample_stop = subsample_config[1]
        subsample_step = subsample_config[2]

        # colsample_bytree: subsample ratio of columns when constructing each tree.
        colsample_bytree_config = train_config_xgb[
            "gridsearch_params_start_stop_step"
        ].get("subsample", [6, 11, 2])
        colsample_bytree_start = colsample_bytree_config[0]
        colsample_bytree_stop = colsample_bytree_config[1]
        colsample_bytree_step = colsample_bytree_config[2]

        # confusion matrix plotting parameters:
        cm_include_count = train_config_xgb["plot_params"].get(
            "cm_include_count", False
        )
        cm_include_percent = train_config_xgb["plot_params"].get(
            "cm_include_percent", True
        )
        annotate_scores = train_config_xgb["plot_params"].get("annotate_scores", False)

        # seed: random seed
        seed = random_state

        # nfold: number of folds during cross-validation
        nfold = train_config_xgb["other_training_params"].get("nfold", 5)

        # metrics: evaluation metrics to use during cross-validation
        metrics = train_config_xgb["other_training_params"].get("metrics", ["auc"])

        # objective: name of learning objective
        objective = train_config_xgb["other_training_params"].get(
            "objective", "binary:logistic"
        )

        # eval_metric: Evaluation metrics for validation data
        eval_metric = train_config_xgb["other_training_params"].get(
            "eval_metric", "auc"
        )

        # early_stopping_rounds: Validation metric needs to improve at least once in every early_stopping_rounds round(s) to continue training
        early_stopping_rounds = train_config_xgb["other_training_params"].get(
            "early_stopping_rounds", 10
        )

        # num_boost_round: Number of boosting iterations
        num_boost_round = train_config_xgb["other_training_params"].get(
            "num_boost_round", 999
        )

        # parse boolean args
        dense_branch = forgiving_true(dense_branch)
        conv_branch = forgiving_true(conv_branch)
        run_eagerly = forgiving_true(run_eagerly)
        save = forgiving_true(save)
        plot = forgiving_true(plot)
        cm_include_count = forgiving_true(cm_include_count)
        cm_include_percent = forgiving_true(cm_include_percent)
        annotate_scores = forgiving_true(annotate_scores)

        time_tag = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        output_path = self.base_path / f"models_{algorithm}" / group

        if algorithm == "dnn":

            import tensorflow as tf

            if gpu is not None:
                gpus = tf.config.list_physical_devices("GPU")
                tf.config.experimental.set_visible_devices(gpus[gpu], "GPU")
            else:
                tf.config.experimental.set_visible_devices([], "GPU")

            import wandb
            from wandb.keras import WandbCallback

            from .nn import DNN

            classifier = DNN(name=tag)

            if run_sweeps:
                # Point to datasets needed for training/validation
                classifier.assign_datasets(
                    features_input_shape=len(features),
                    train_dataset_repeat=datasets["train_repeat"],
                    val_dataset_repeat=datasets["val_repeat"],
                    steps_per_epoch_train=steps_per_epoch["train"],
                    steps_per_epoch_val=steps_per_epoch["val"],
                    train_dataset=datasets["train"],
                    val_dataset=datasets["val"],
                    wandb_token=self.config["wandb"]["token"],
                )

                wandb.login(key=self.config["wandb"]["token"])

                # Define sweep config
                sweep_configuration = self.config["wandb"]["sweep_config_dnn"]
                sweep_configuration["name"] = f"{group}-{tag}-{time_tag}"

                entity = self.config["wandb"]["entity"]
                project = self.config["wandb"]["project"]

                # Set up sweep/id
                sweep_id = wandb.sweep(
                    sweep=sweep_configuration,
                    project=project,
                )

                # Start sweep job
                wandb.agent(sweep_id, function=classifier.sweep)

                print(
                    "Sweep complete. Adjust hyperparameters in config file and run scope-train again without the --run-sweeps flag."
                )

                # Stop sweep job
                try:
                    print("Stopping sweep.")
                    os.system(
                        f"python -m wandb sweep --stop {entity}/{project}/{sweep_id}"
                    )
                except Exception:
                    print("Sweep already stopped.")

                return

            if pre_trained_model is not None:
                classifier.load(pre_trained_model, weights_only=weights_only)
                model_input = classifier.model.input
                training_set_inputs = datasets["train"].element_spec[0]
                # Compare input shapes with model inputs
                print(
                    "Comparing shapes of input features with inputs for existing model..."
                )
                for inpt in model_input:
                    inpt_name = inpt.name
                    inpt_shape = inpt.shape
                    inpt_shape.assert_is_compatible_with(
                        training_set_inputs[inpt_name].shape
                    )
                print("Input shapes are consistent.")
                classifier.set_callbacks(callbacks, tag, **kwargs)

            else:
                classifier.setup(
                    dense_branch=dense_branch,
                    features_input_shape=(len(features),),
                    conv_branch=conv_branch,
                    dmdt_input_shape=(26, 26, 1),
                    loss=loss,
                    optimizer=optimizer,
                    learning_rate=lr,
                    momentum=momentum,
                    monitor=monitor,
                    patience=patience,
                    callbacks=callbacks,
                    run_eagerly=run_eagerly,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    epsilon=epsilon,
                    decay=decay,
                    amsgrad=amsgrad,
                )

            if plot_model:
                tf.keras.utils.plot_model(
                    classifier.model,
                    to_file=self.base_path / "DNN.pdf",
                    show_shapes=True,
                    show_dtype=False,
                    show_layer_names=False,
                    rankdir="TB",
                    expand_nested=False,
                    dpi=300,
                    layer_range=None,
                    show_layer_activations=True,
                    show_trainable=False,
                )

            if verbose:
                print(classifier.model.summary())

            if evaluate_test:
                wandb.login(key=self.config["wandb"]["token"])
                wandb.init(
                    job_type=job_type,
                    project=self.config["wandb"]["project"],
                    tags=[tag],
                    group=group,
                    name=f"{tag}-{time_tag}",
                    config={
                        "tag": tag,
                        "label": label,
                        "dataset": pathlib.Path(path_dataset).name,
                        "scale_features": scale_features,
                        "learning_rate": lr,
                        "epochs": epochs,
                        "patience": patience,
                        "random_state": random_state,
                        "batch_size": batch_size,
                        "architecture": "scope-net",
                        "dense_branch": dense_branch,
                        "conv_branch": conv_branch,
                    },
                )
                classifier.meta["callbacks"].append(WandbCallback())

                classifier.train(
                    datasets["train_repeat"],
                    datasets["val_repeat"],
                    steps_per_epoch["train"],
                    steps_per_epoch["val"],
                    epochs=epochs,
                    class_weight=class_weight,
                    verbose=verbose,
                )

        elif algorithm == "xgb":

            # XGB-specific code
            X_train = ds.df_ds.loc[indexes["train"]][features]
            y_train = ds.target[indexes["train"]]

            X_val = ds.df_ds.loc[indexes["val"]][features]
            y_val = ds.target[indexes["val"]]

            X_test = ds.df_ds.loc[indexes["test"]][features]
            y_test = ds.target[indexes["test"]]

            scale_pos_weight = class_weight[1] / class_weight[0]

            # Add code to train XGB algorithm
            classifier = XGB(name=tag)
            classifier.setup(
                max_depth=max_depth_start,
                min_child_weight=min_child_weight_start,
                eta=eta_list[0],
                subsample=subsample_start / 10.0,
                colsample_bytree=colsample_bytree_start / 10.0,
                objective=objective,
                eval_metric=eval_metric,
                early_stopping_rounds=early_stopping_rounds,
                num_boost_round=num_boost_round,
                scale_pos_weight=scale_pos_weight,
            )
            classifier.train(
                X_train,
                y_train,
                X_val,
                y_val,
                skip_cv=skip_cv,
                seed=seed,
                nfold=nfold,
                metrics=metrics,
                max_depth_start=max_depth_start,
                max_depth_stop=max_depth_stop,
                max_depth_step=max_depth_step,
                min_child_weight_start=min_child_weight_start,
                min_child_weight_stop=min_child_weight_stop,
                min_child_weight_step=min_child_weight_step,
                eta_list=eta_list,
                subsample_start=subsample_start,
                subsample_stop=subsample_stop,
                subsample_step=subsample_step,
                colsample_bytree_start=colsample_bytree_start,
                colsample_bytree_stop=colsample_bytree_stop,
                colsample_bytree_step=colsample_bytree_step,
            )

        if verbose:
            print("Evaluating on train/val/test sets:")
        # TODO: there should not need to be this algorithm-based split in the call to classifier.evaluate()
        if algorithm == "xgb":
            stats_train = classifier.evaluate(X_train, y_train, name="train")
            stats_val = classifier.evaluate(X_val, y_val, name="val")
            stats_test = classifier.evaluate(X_test, y_test, name="test")
        else:
            stats_train = classifier.evaluate(
                datasets["train"], name="train", verbose=verbose
            )
            stats_val = classifier.evaluate(
                datasets["val"], name="val", verbose=verbose
            )
            stats_test = classifier.evaluate(
                datasets["test"], name="test", verbose=verbose
            )

        print("training stats: ", stats_train)
        print("validation stats: ", stats_val)
        if verbose:
            print("test stats: ", stats_test)

        if algorithm == "DNN":
            param_names = (
                "loss",
                "tp",
                "fp",
                "tn",
                "fn",
                "accuracy",
                "precision",
                "recall",
                "auc",
            )
            if evaluate_test:
                # log model performance on the test set
                for param, value in zip(param_names, stats_test):
                    wandb.run.summary[f"test_{param}"] = value
                p, r = (
                    wandb.run.summary["test_precision"],
                    wandb.run.summary["test_recall"],
                )
                wandb.run.summary["test_f1"] = (
                    2 * p * r / (p + r) if (p + r) > 0 else 0.0
                )

            if datasets["dropped_samples"] is not None:
                # log model performance on the dropped samples
                if verbose:
                    print("Evaluating on samples dropped from the training set:")
                stats = classifier.evaluate(
                    datasets["dropped_samples"], verbose=verbose
                )
                if verbose:
                    print(stats)

                if evaluate_test:
                    for param, value in zip(param_names, stats):
                        wandb.run.summary[f"dropped_samples_{param}"] = value
                    p, r = (
                        wandb.run.summary["dropped_samples_precision"],
                        wandb.run.summary["dropped_samples_recall"],
                    )
                    wandb.run.summary["dropped_samples_f1"] = (
                        2 * p * r / (p + r) if (p + r) > 0 else 0.0
                    )

        if save:
            if verbose:
                print(f"Saving model to {output_path / tag}")
            classifier.save(
                output_path=str(output_path / tag),
                tag=f"{tag}.{time_tag}",
                plot=plot,
                cm_include_count=cm_include_count,
                cm_include_percent=cm_include_percent,
                annotate_scores=annotate_scores,
            )

            return time_tag

    def parse_run_create_training_script(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--filename",
            type=str,
            default="train_script.sh",
            help="filename of shell script (must not currently exist)",
        )
        parser.add_argument(
            "--algorithm",
            type=str,
            default="dnn",
            help="name of algorithm to use for training",
        )
        parser.add_argument(
            "--min-count",
            type=int,
            default=100,
            help="minimum number of positive examples to include in script",
        )
        parser.add_argument(
            "--path-dataset",
            type=str,
            help="local path to .parquet, .h5 or .csv file with the dataset, if not provided in config.yaml",
        )
        parser.add_argument(
            "--pre-trained-group-name",
            type=str,
            help="name of group containing pre-trained models within models directory",
        )
        parser.add_argument(
            "--add-keywords",
            type=str,
            default="",
            help="str containing additional training keywords to append to each line in the script",
        )
        parser.add_argument(
            "--train-all",
            action="store_true",
            help="if group_name is specified, set this keyword to train all classes regardeless of whether a trained model exists",
        )
        parser.add_argument(
            "--period-suffix",
            type=str,
            help="suffix of period/Fourier features to use for training",
        )

        args, _ = parser.parse_known_args()
        self.create_training_script(**vars(args))

    def create_training_script(
        self,
        filename: str = "train_script.sh",
        algorithm: str = "dnn",
        min_count: int = 100,
        path_dataset: str = None,
        pre_trained_group_name: str = None,
        add_keywords: str = "",
        train_all: bool = False,
        period_suffix: str = None,
    ):
        """
        Create training shell script from classes in config file meeting minimum count requirement

        :param filename: filename of shell script (must not currently exist) (str)
        :param algorithm: name of algorithm to use for training (str)
        :param min_count: minimum number of positive examples to include in script (int)
        :param path_dataset: local path to .parquet, .h5 or .csv file with the dataset, if not provided in config.yaml (str)
        :param pre_trained_group_name: name of group containing pre-trained models within models directory (str)
        :param add_keywords: str containing additional training keywords to append to each line in the script
        :param train_all: if group_name is specified, set this keyword to train all classes regardeless of whether a trained model exists (bool)
        :param period_suffix: suffix of period/Fourier features to use for training (str)

        :return:

        :examples:  create-training-script --filename train_dnn.sh --algorithm dnn --min-count 1000 \
                    --path-dataset tools/fritzDownload/merged_classifications_features.parquet --add-keywords '--save --plot --group groupname'

                    create-training-script --filename train_xgb.sh --algorithm xgb --min-count 100 \
                    --add-keywords '--save --plot --batch-size 32 --group groupname'
        """
        path = str(self.base_path / filename)

        phenom_tags = []
        ontol_tags = []

        if period_suffix is None:
            period_suffix = self.config["features"]["info"]["period_suffix"]

        if path_dataset is None:
            dataset_name = self.config["training"]["dataset"]
            path_dataset = str(self.base_path / dataset_name)

        if path_dataset.endswith(".parquet"):
            dataset = read_parquet(path_dataset)
        elif path_dataset.endswith(".h5"):
            dataset = read_hdf(path_dataset)
        elif path_dataset.endswith(".csv"):
            dataset = pd.read_csv(path_dataset)
        else:
            raise ValueError(
                "Dataset in config file must end with .parquet, .h5 or .csv"
            )

        with open(path, "x") as script:

            script.write("#!/bin/bash\n")

            for tag in self.config["training"]["classes"].keys():
                label = self.config["training"]["classes"][tag]["label"]
                threshold = self.config["training"]["classes"][tag]["threshold"]
                branch = self.config["training"]["classes"][tag]["features"]
                num_pos = np.sum(dataset[label] > threshold)

                if num_pos > min_count:
                    print(
                        f"Label {label}: {num_pos} positive examples with P > {threshold}"
                    )
                    if branch == "phenomenological":
                        phenom_tags += [tag]
                    else:
                        ontol_tags += [tag]

            if pre_trained_group_name is not None:
                group_path = (
                    self.base_path / f"models_{algorithm}" / pre_trained_group_name
                )
                gen = os.walk(group_path)
                model_tags = [tag[1] for tag in gen]
                model_tags = model_tags[0]

                # If a group name for trained models is provided, either train all classes in config or only those with existing trained model
                phenom_hasmodel = list(
                    set.intersection(set(phenom_tags), set(model_tags))
                )
                ontol_hasmodel = list(
                    set.intersection(set(ontol_tags), set(model_tags))
                )

                script.write("# Phenomenological\n")
                for tag in phenom_tags:
                    if tag in phenom_hasmodel:
                        tag_file_gen = (group_path / tag).glob("*.h5")
                        most_recent_file = max(
                            [file for file in tag_file_gen], key=os.path.getctime
                        ).name

                        script.writelines(
                            f"scope-train --tag {tag} --algorithm {algorithm} --path_dataset {path_dataset} --pre_trained_model models/{pre_trained_group_name}/{tag}/{most_recent_file} --period_suffix {period_suffix} --verbose {add_keywords} \n"
                        )

                    elif train_all:
                        script.writelines(
                            f"scope-train --tag {tag} --algorithm {algorithm} --path_dataset {path_dataset} --period_suffix {period_suffix} --verbose {add_keywords} \n"
                        )

                script.write("# Ontological\n")
                for tag in ontol_tags:
                    if tag in ontol_hasmodel:
                        tag_file_gen = (group_path / tag).glob("*.h5")
                        most_recent_file = max(
                            [file for file in tag_file_gen], key=os.path.getctime
                        ).name

                        script.writelines(
                            f"scope-train --tag {tag} --algorithm {algorithm} --path_dataset {path_dataset} --pre_trained_model models/{pre_trained_group_name}/{tag}/{most_recent_file} --period_suffix {period_suffix} --verbose {add_keywords} \n"
                        )

                    elif train_all:
                        script.writelines(
                            f"scope-train --tag {tag} --algorithm {algorithm} --path_dataset {path_dataset} --period_suffix {period_suffix} --verbose {add_keywords} \n"
                        )

            else:
                script.write("# Phenomenological\n")
                script.writelines(
                    [
                        f"scope-train --tag {tag} --algorithm {algorithm} --path_dataset {path_dataset} --period_suffix {period_suffix} --verbose {add_keywords} \n"
                        for tag in phenom_tags
                    ]
                )
                script.write("# Ontological\n")
                script.writelines(
                    [
                        f"scope-train --tag {tag} --algorithm {algorithm} --path_dataset {path_dataset} --period_suffix {period_suffix} --verbose {add_keywords} \n"
                        for tag in ontol_tags
                    ]
                )
        print(f"Wrote traininig script to {path}.")

    def parse_run_assemble_training_stats(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--group-name",
            type=str,
            default="experiment",
            help="trained model group name",
        )
        parser.add_argument(
            "--algorithm",
            type=str,
            default="dnn",
            help="name of ML algorithm",
        )
        parser.add_argument(
            "--set-name",
            type=str,
            default="val",
            help="one of train, val or test",
        )
        parser.add_argument(
            "--importance-directory",
            type=str,
            default="xgb_feature_importance",
            help="name of directory to save XGB feature importance",
        )
        parser.add_argument(
            "--stats-directory",
            type=str,
            default="stats",
            help="name of directory to save training stats",
        )

        args, _ = parser.parse_known_args()
        self.assemble_training_stats(**vars(args))

    def assemble_training_stats(
        self,
        group_name: str = "experiment",
        algorithm: str = "dnn",
        set_name: str = "val",
        importance_directory: str = "xgb_feature_importance",
        stats_directory: str = "stats",
    ):
        """
        Assemble training stats from individal class results

        :param group_name: trained model group name (str)
        :param algorithm: name of ML algorithm (str)
        :param set_name: one of train, val or test (str)
        :param importance_directory: name of directory to save XGB feature importance (str)
        :param stats_directory: name of directory to save training stats (str)

        :return:

        :example: assemble-training-stats --group-name DR16 --algorithm xgb --set-name test \
                  --importance-directory xgb_importance --stats-directory xgb_stats
        """
        base_path = self.base_path
        group_path = base_path / f"models_{algorithm}" / group_name

        if algorithm in ["xgb", "xgboost", "XGB", "XGBoost"]:
            importance_path = base_path / importance_directory
            importance_path.mkdir(exist_ok=True)

            # XGB feature importance
            labels = [x for x in group_path.iterdir() if x.name != ".DS_Store"]
            statpaths = []
            for label in labels:
                statpaths.append(
                    [x for x in label.glob(f"*plots/{set_name}/*impvars.json")][0]
                )

            for statpath in statpaths:
                strpath = str(statpath)
                os.system(f"cp {strpath} {importance_path}/.")

        # DNN/XGB stats
        stats_path = base_path / f"{algorithm}_{stats_directory}"
        stats_path.mkdir(exist_ok=True)
        labels = [x for x in group_path.iterdir() if x.name != ".DS_Store"]
        statpaths = []
        for label in labels:
            statpaths.append(
                [x for x in label.glob(f"*plots/{set_name}/*stats.json")][0]
            )

        for statpath in statpaths:
            strpath = str(statpath)
            os.system(f"cp {strpath} {stats_path}/.")

        print("Finished assembling stats.")

    def parse_run_create_inference_script(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--filename",
            type=str,
            default="get_all_preds_dnn.sh",
            help="filename of shell script (must not currently exist)",
        )
        parser.add_argument(
            "--group-name",
            type=str,
            default="experiment",
            help="name of group containing trained models within models directory",
        )
        parser.add_argument(
            "--algorithm",
            type=str,
            default="dnn",
            help="algorithm to use in script",
        )
        parser.add_argument(
            "--scale-features",
            type=str,
            default="min_max",
            help="method to scale features (currently 'min_max' or 'median_std')",
        )
        parser.add_argument(
            "--feature-directory",
            type=str,
            default="features",
            help="name of directory containing downloaded or generated features",
        )
        parser.add_argument(
            "--write-csv",
            action="store_true",
            help="if set, write CSV file in addition to parquet",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=100000,
            help="batch size to use when reading feature files",
        )
        parser.add_argument(
            "--use-custom-python",
            action="store_true",
            help="if True, the call to run-inference will be preceded by a specific path to python",
        )
        parser.add_argument(
            "--path-to-python",
            type=str,
            default="~/miniforge3/envs/scope-env/bin/python",
            help="if --use-custom-python is set (e.g. for a cron job), path to custom python installation",
        )
        parser.add_argument(
            "--period-suffix",
            type=str,
            help="suffix of period/Fourier features to use for training",
        )

        args, _ = parser.parse_known_args()
        self.create_inference_script(**vars(args))

    def create_inference_script(
        self,
        filename: str = "get_all_preds_dnn.sh",
        group_name: str = "experiment",
        algorithm: str = "dnn",
        scale_features: str = "min_max",
        feature_directory: str = "features",
        write_csv: bool = False,
        batch_size: int = 100000,
        use_custom_python: bool = False,
        path_to_python: str = "~/miniforge3/envs/scope-env/bin/python",
        period_suffix: str = None,
    ):
        """
        Save shell script to use when running inference

        :param filename: filename of shell script (must not currently exist) (str)
        :param group_name: name of group containing trained models within models directory (str)
        :param algorithm: algorithm to use in script (str)
        :param scale_features: method to scale features (str, currently "min_max" or "median_std")
        :param feature_directory: name of directory containing downloaded or generated features (str)
        :param write_csv: if True, write CSV file in addition to parquet (bool)
        :param batch_size: batch size to use when reading feature files (int)
        :param use_custom_python: if True, the call to run-inference will be preceded by a specific path to python (bool)
        :param path_to_python: if use_custom_python is set (e.g. for a cron job), path to custom python installation (str)
        :param period_suffix: suffix of period/Fourier features to use for training (str)

        :return:

        :example:  create-inference-script --filename get_all_preds_dnn.sh --group-name experiment \
                    --algorithm dnn --feature-directory generated_features
        """
        base_path = self.base_path
        path = str(base_path / filename)
        group_path = base_path / f"models_{algorithm}" / group_name

        addtl_args = ""
        if write_csv:
            addtl_args += "--write-csv"

        gen = os.walk(group_path)
        model_tags = [tag[1] for tag in gen]
        model_tags = model_tags[0]

        if period_suffix is None:
            period_suffix = self.config["features"]["info"]["period_suffix"]

        if not use_custom_python:
            path_to_python = ""

        with open(path, "x") as script:
            script.write("#!/bin/bash\n")
            script.write(
                "# Call script followed by field number, e.g: ./get_all_preds_dnn.sh 301\n"
            )

            paths_models_str = ""
            model_class_names_str = ""

            if algorithm in ["dnn", "DNN", "nn", "NN"]:
                algorithm = "dnn"
                script.write('echo "dnn inference"\n')
                # Select most recent model for each tag
                for tag in model_tags:
                    tag_file_gen = (group_path / tag).glob("*.h5")
                    most_recent_file = max(
                        [file for file in tag_file_gen], key=os.path.getctime
                    ).name

                    paths_models_str += f"{str(base_path)}/models_{algorithm}/{group_name}/{tag}/{most_recent_file} "
                    model_class_names_str += f"{tag} "

                script.write(
                    f'echo -n "Running inference..." && {path_to_python} run-inference --paths-models {paths_models_str} --model-class-names {model_class_names_str} --field $1 --whole-field --flag-ids --scale-features {scale_features} --feature-directory {feature_directory} --period-suffix {period_suffix} --batch-size {batch_size} {addtl_args} && echo "done"\n'
                )

            elif algorithm in ["XGB", "xgb", "XGBoost", "xgboost", "XGBOOST"]:
                algorithm = "xgb"
                script.write('echo "xgb inference"\n')
                for tag in model_tags:
                    tag_file_gen = (group_path / tag).glob("*.json")
                    most_recent_file = max(
                        [file for file in tag_file_gen], key=os.path.getctime
                    ).name

                    paths_models_str += f"{str(base_path)}/models_{algorithm}/{group_name}/{tag}/{most_recent_file} "
                    model_class_names_str += f"{tag} "

                script.write(
                    f'echo -n "Running inference..." && {path_to_python} run-inference --paths-models {paths_models_str} --model-class-names {model_class_names_str} --scale-features {scale_features} --feature-directory {feature_directory} --period-suffix {period_suffix} --batch-size {batch_size} --xgb-model --field $1 --whole-field --flag-ids {addtl_args} && echo "done"\n'
                )

            else:
                raise ValueError("algorithm must be dnn or xgb")

        print(f"Wrote inference script to {path}")

    def consolidate_inference_results(
        self,
        dataset: pd.DataFrame,
        statistic: str = "mean",
    ):
        """
        Consolidate inference results from multiple rows to one per source (called in select_fritz_sample)

        :param dataset: inference results from 'preds' directory (pandas DataFrame)
        :param statistic: method to combine multiple predictions for single source [mean, median and max currently supported] (str)

        """

        # Define subsets of data with or without Gaia, AllWISE and PS1 IDs.
        # Survey IDs are used to identify unique sources.
        # Begin with Gaia EDR3 ID
        # If no Gaia ID, use AllWISE
        # If no AllWISE, use PS1
        withGaiaID = dataset[dataset["Gaia_EDR3___id"] != 0].reset_index(drop=True)
        nanGaiaID = dataset[dataset["Gaia_EDR3___id"] == 0].reset_index(drop=True)

        withAllWiseID = nanGaiaID[nanGaiaID["AllWISE___id"] != 0].reset_index(drop=True)
        nanAllWiseID = nanGaiaID[nanGaiaID["AllWISE___id"] == 0].reset_index(drop=True)

        withPS1ID = nanAllWiseID[nanAllWiseID["PS1_DR1___id"] != 0].reset_index(
            drop=True
        )

        # Define columns for each subset that should not be averaged or otherwise aggregated

        skipList = ["Gaia_EDR3___id", "AllWISE___id", "PS1_DR1___id", "_id"]

        skip_mean_cols_Gaia = withGaiaID[skipList]
        skip_mean_cols_AllWise = withAllWiseID[skipList]
        skip_mean_cols_PS1 = withPS1ID[skipList]

        if statistic in [
            "mean",
            "Mean",
            "MEAN",
            "average",
            "AVERAGE",
            "Average",
            "avg",
            "AVG",
        ]:
            groupedMeans_Gaia = (
                withGaiaID.groupby("Gaia_EDR3___id")
                .mean()
                .drop(["_id", "AllWISE___id", "PS1_DR1___id"], axis=1)
                .reset_index()
            )

            groupedMeans_AllWise = (
                withAllWiseID.groupby("AllWISE___id")
                .mean()
                .drop(["_id", "Gaia_EDR3___id", "PS1_DR1___id"], axis=1)
                .reset_index()
            )

            groupedMeans_PS1 = (
                withPS1ID.groupby("PS1_DR1___id")
                .mean()
                .drop(["_id", "Gaia_EDR3___id", "AllWISE___id"], axis=1)
                .reset_index()
            )

        elif statistic in ["max", "Max", "MAX", "maximum", "Maximum", "MAXIMUM"]:
            groupedMeans_Gaia = (
                withGaiaID.groupby("Gaia_EDR3___id")
                .max()
                .drop(["_id", "AllWISE___id", "PS1_DR1___id"], axis=1)
                .reset_index()
            )

            groupedMeans_AllWise = (
                withAllWiseID.groupby("AllWISE___id")
                .max()
                .drop(["_id", "Gaia_EDR3___id", "PS1_DR1___id"], axis=1)
                .reset_index()
            )

            groupedMeans_PS1 = (
                withPS1ID.groupby("PS1_DR1___id")
                .max()
                .drop(["_id", "Gaia_EDR3___id", "AllWISE___id"], axis=1)
                .reset_index()
            )

        elif statistic in ["median", "Median", "MEDIAN", "med", "MED"]:
            groupedMeans_Gaia = (
                withGaiaID.groupby("Gaia_EDR3___id")
                .median()
                .drop(["_id", "AllWISE___id", "PS1_DR1___id"], axis=1)
                .reset_index()
            )

            groupedMeans_AllWise = (
                withAllWiseID.groupby("AllWISE___id")
                .median()
                .drop(["_id", "Gaia_EDR3___id", "PS1_DR1___id"], axis=1)
                .reset_index()
            )

            groupedMeans_PS1 = (
                withPS1ID.groupby("PS1_DR1___id")
                .median()
                .drop(["_id", "Gaia_EDR3___id", "AllWISE___id"], axis=1)
                .reset_index()
            )

        else:
            raise ValueError(
                "Mean, median and max are the currently supported statistics."
            )

        # Construct new survey_id column that contains the ID used to add grouped source to the list
        string_ids_Gaia = groupedMeans_Gaia["Gaia_EDR3___id"].astype(str)
        groupedMeans_Gaia["survey_id"] = ["Gaia_EDR3___" + s for s in string_ids_Gaia]

        string_ids_AllWise = groupedMeans_AllWise["AllWISE___id"].astype(str)
        groupedMeans_AllWise["survey_id"] = [
            "AllWISE___" + s for s in string_ids_AllWise
        ]

        string_ids_PS1 = groupedMeans_PS1["PS1_DR1___id"].astype(str)
        groupedMeans_PS1["survey_id"] = ["PS1_DR1___" + s for s in string_ids_PS1]

        # Merge averaged, non-averaged columns on obj_id
        allRows_Gaia = pd.merge(
            groupedMeans_Gaia, skip_mean_cols_Gaia, on=["Gaia_EDR3___id"]
        )
        noDup_ids_Gaia = allRows_Gaia.drop_duplicates("Gaia_EDR3___id")[
            ["Gaia_EDR3___id", "_id"]
        ]
        groupedMeans_Gaia = pd.merge(
            groupedMeans_Gaia, noDup_ids_Gaia, on="Gaia_EDR3___id"
        )
        groupedMeans_Gaia.drop("Gaia_EDR3___id", axis=1, inplace=True)

        allRows_AllWise = pd.merge(
            groupedMeans_AllWise, skip_mean_cols_AllWise, on=["AllWISE___id"]
        )
        noDup_ids_AllWise = allRows_AllWise.drop_duplicates("AllWISE___id")[
            ["AllWISE___id", "_id"]
        ]
        groupedMeans_AllWise = pd.merge(
            groupedMeans_AllWise, noDup_ids_AllWise, on="AllWISE___id"
        )
        groupedMeans_AllWise.drop("AllWISE___id", axis=1, inplace=True)

        allRows_PS1 = pd.merge(
            groupedMeans_PS1, skip_mean_cols_PS1, on=["PS1_DR1___id"]
        )
        noDup_ids_PS1 = allRows_PS1.drop_duplicates("PS1_DR1___id")[
            ["PS1_DR1___id", "_id"]
        ]
        groupedMeans_PS1 = pd.merge(groupedMeans_PS1, noDup_ids_PS1, on="PS1_DR1___id")
        groupedMeans_PS1.drop("PS1_DR1___id", axis=1, inplace=True)

        # Create dataframe with one row per source
        consol_rows = pd.concat(
            [groupedMeans_Gaia, groupedMeans_AllWise, groupedMeans_PS1]
        ).reset_index(drop=True)

        # Create dataframe containing all rows (including duplicates for multiple light curves)
        all_rows = pd.concat([allRows_Gaia, allRows_AllWise, allRows_PS1])
        all_rows.drop(
            ["Gaia_EDR3___id", "AllWISE___id", "PS1_DR1___id"], axis=1, inplace=True
        )

        # Reorder columns for better legibility
        consol_rows = consol_rows.set_index("survey_id").reset_index()
        all_rows = all_rows.set_index("survey_id").reset_index()

        return consol_rows, all_rows

    def parse_run_select_fritz_sample(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--fields",
            type=Union[int, str],
            nargs="+",
            default=["all"],
            help="list of field predictions (integers) to include, 'all' to use all available fields, or 'specific_ids' if running on e.g. GCN sources",
        )
        parser.add_argument(
            "--group",
            type=str,
            default="experiment",
            help="name of group containing trained models within models directory",
        )
        parser.add_argument(
            "--min-class-examples",
            type=int,
            default=1000,
            help="minimum number of examples to include for each class. Some classes may contain fewer than this if the sample is limited",
        )
        parser.add_argument(
            "--select-top-n",
            action="store_true",
            help="if set, select top N probabilities above probability_threshold from each class",
        )
        parser.add_argument(
            "--include-all-highprob-labels",
            action="store_true",
            help="if select_top_n is set, setting this keyword includes any classification above the probability_threshold for all top N sources. Otherwise, literally only the top N probabilities for each classification will be included, which may artifically exclude relevant labels.",
        )
        parser.add_argument(
            "--probability-threshold",
            type=float,
            default=0.9,
            help="minimum probability to select for Fritz",
        )
        parser.add_argument(
            "--al-directory",
            type=str,
            default="AL_datasets",
            help="name of directory to create/populate with Fritz sample",
        )
        parser.add_argument(
            "--al-filename",
            type=str,
            default="active_learning_set",
            help="name of file (no extension) to store Fritz sample",
        )
        parser.add_argument(
            "--algorithm",
            type=str,
            default="dnn",
            help="ML algorithm (dnn or xgb)",
        )
        parser.add_argument(
            "--exclude-training-sources",
            action="store_true",
            help="if set, exclude sources in current training set from AL sample",
        )
        parser.add_argument(
            "--write-csv",
            action="store_true",
            help="if set, write CSV file in addition to parquet",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="if set, print additional information",
        )
        parser.add_argument(
            "--consolidation-statistic",
            type=str,
            default="mean",
            help="method to combine multiple classification probabilities for a single source ('mean', 'median' or 'max' currently supported)",
        )
        parser.add_argument(
            "--read-consolidation-results",
            action="store_true",
            help="if set, search for and read an existing consolidated file having _consol.parquet suffix",
        )
        parser.add_argument(
            "--write-consolidation-results",
            action="store_true",
            help="if set, save two files: consolidated inference results [1 row per source] and full results [≥ 1 row per source]",
        )
        parser.add_argument(
            "--consol-filename",
            type=str,
            default="inference_results",
            help="name of file (no extension) to store consolidated and full results",
        )
        parser.add_argument(
            "--doNotSave",
            action="store_true",
            help="if set, do not write results",
        )
        parser.add_argument(
            "--doAllSources",
            action="store_true",
            help="if set, ignore min_class_examples and run for all sources",
        )

        args, _ = parser.parse_known_args()
        self.select_fritz_sample(**vars(args))

    def select_fritz_sample(
        self,
        fields: list = ["all"],
        group: str = "experiment",
        min_class_examples: int = 1000,
        select_top_n: bool = False,
        include_all_highprob_labels: bool = False,
        probability_threshold: float = 0.9,
        al_directory: str = "AL_datasets",
        al_filename: str = "active_learning_set",
        algorithm: str = "dnn",
        exclude_training_sources: bool = False,
        write_csv: bool = True,
        verbose: bool = False,
        consolidation_statistic: str = "mean",
        read_consolidation_results: bool = False,
        write_consolidation_results: bool = False,
        consol_filename: str = "inference_results",
        doNotSave: bool = False,
        doAllSources: bool = False,
    ):
        """
        Select subset of predictions to use for posting to Fritz (active learning, GCN source classifications).

        :param fields: list of field predictions (integers) to include, 'all' to use all available fields, or 'specific_ids' if running on e.g. GCN sources (list)
        :param group: name of group containing trained models within models directory (str)
        :param min_class_examples: minimum number of examples to include for each class. Some classes may contain fewer than this if the sample is limited (int)
        :param select_top_n: if set, select top N probabilities above probability_threshold from each class (bool)
        :param include_all_highprob_labels: if select_top_n is set, setting this keyword includes any classification above the probability_threshold for all top N sources.
            Otherwise, literally only the top N probabilities for each classification will be included, which may artifically exclude relevant labels.
        :param probability_threshold: minimum probability to select for Fritz (float)
        :param al_directory: name of directory to create/populate with Fritz sample (str)
        :param al_filename: name of file (no extension) to store Fritz sample (str)
        :param algorithm: algorithm [dnn or xgb] (str)
        :param exclude_training_sources: if True, exclude sources in current training set from AL sample (bool)
        :param write_csv: if set, write CSV file in addition to parquet (bool)
        :param verbose: if set, print additional information (bool)
        :param consolidation_statistic: method to combine multiple classification probabilities for a single source [mean, median or max currently supported] (str)
        :param read_consolidation_results: if set, search for and read an existing consolidated file having _consol.parquet suffix (bool)
        :param write_consolidation_results: if set, save two files: consolidated inference results [1 row per source] and full results [≥ 1 row per source] (bool)
        :param consol_filename: name of file (no extension) to store consolidated and full results (str)
        :param doNotSave: if set, do not write results (bool)
        :param doAllSources: if set, ignore min_class_examples and run for all sources (bool)

        :return:
        final_toPost: DataFrame containing sources with high-confidence classifications to post

        :examples:  select-fritz-sample --fields 296 297 --group experiment --min-class-examples 1000 --probability-threshold 0.9 --exclude-training-sources --write-consolidation-results
                    select-fritz-sample --fields 296 297 --group experiment --min-class-examples 500 --select-top-n --include-all-highprob-labels --probability-threshold 0.7 --exclude-training-sources --read-consolidation-results
                    select-fritz-sample --fields specific_ids --group DR16 --algorithm xgb --probability-threshold 0.9 --consol-filename inference_results_specific_ids --al-directory=GCN --al-filename GCN_sources --write-consolidation-results --select-top-n --doAllSources --write-csv

        """
        base_path = self.base_path
        if algorithm in ["DNN", "NN", "dnn", "nn"]:
            algorithm = "dnn"
        elif algorithm in ["XGB", "xgb", "XGBoost", "xgboost", "XGBOOST"]:
            algorithm = "xgb"
        else:
            raise ValueError("Algorithm must be either dnn or xgb.")

        preds_path = base_path / f"preds_{algorithm}"

        # Strip extension from filename if provided
        al_filename = al_filename.split(".")[0]
        AL_directory_path = str(base_path / f"{al_directory}_{algorithm}" / al_filename)
        os.makedirs(AL_directory_path, exist_ok=True)

        df_coll = []
        df_coll_allRows = []
        if "all" in fields:
            gen_fields = os.walk(preds_path)
            fields = [x for x in gen_fields][0][1]
            print(f"Generating Fritz sample from {len(fields)} fields:")
        elif "specific_ids" in fields:
            fields = [f"field_{fields}"]
            print("Generating Fritz sample from specific ids across multiple fields:")
        else:
            fields = [f"field_{f}" for f in fields]
            print(f"Generating Fritz sample from {len(fields)} fields:")

        column_nums = []

        AL_directory_PL = pathlib.Path(AL_directory_path)
        gen = AL_directory_PL.glob(f"{consol_filename}_consol.parquet")
        existing_consol_files = [str(x) for x in gen]

        if (read_consolidation_results) & (len(existing_consol_files) > 0):
            print("Loading existing consolidated results...")
            preds_df = read_parquet(existing_consol_files[0])

        else:
            print("Consolidating classification probabilities to one per source...")
            for field in fields:
                print(field)
                h = read_parquet(str(preds_path / field / f"{field}.parquet"))

                has_obj_id = False
                if "obj_id" in h.columns:
                    has_obj_id = True
                    id_mapper = (
                        h[["_id", "obj_id"]].set_index("_id").to_dict(orient="index")
                    )
                    h.drop("obj_id", axis=1, inplace=True)

                consolidated_df, all_rows_df = self.consolidate_inference_results(
                    h, statistic=consolidation_statistic
                )

                column_nums += [len(consolidated_df.columns)]
                df_coll += [consolidated_df]
                df_coll_allRows += [all_rows_df]

                if verbose:
                    print(field)
                    print(consolidated_df)
                    print()

                if len(np.unique(column_nums)) > 1:
                    raise ValueError(
                        "Not all predictions have the same number of columns."
                    )

                # Create consolidated dataframe (one row per source)
                preds_df = pd.concat(df_coll, axis=0)

                cols = [x for x in preds_df.columns]
                cols.remove("_id")
                cols.remove("survey_id")
                agg_dct = {c: "mean" for c in cols}

                # One more groupby to combine sources across multiple fields
                preds_df = (
                    preds_df.groupby(["survey_id", "_id"]).agg(agg_dct).reset_index()
                )

                # Create dataframe including all light curves (multiple rows per source)
                preds_df_allRows = pd.concat(df_coll_allRows, axis=0)

                if not has_obj_id:
                    # Generate position-based obj_ids for Fritz
                    raArr = [ra for ra in preds_df["ra"]]
                    decArr = [dec for dec in preds_df["dec"]]
                    obj_ids = [radec_to_iau_name(x, y) for x, y in zip(raArr, decArr)]
                else:
                    obj_ids = []
                    for ID in preds_df["_id"]:
                        obj_ids += [id_mapper[ID]["obj_id"]]

                preds_df["obj_id"] = obj_ids

                # Assign obj_ids to all rows
                preds_df_allRows = pd.merge(
                    preds_df_allRows, preds_df[["obj_id", "survey_id"]], on="survey_id"
                )

                # Drop sources which are so close that they cannot be resolved by our position-based ID (~0.0004 of sources)
                preds_df_allRows = (
                    preds_df_allRows.set_index("obj_id")
                    .drop(preds_df[preds_df.duplicated("obj_id")]["obj_id"])
                    .reset_index()
                )
                preds_df = preds_df.drop_duplicates("obj_id", keep=False).reset_index(
                    drop=True
                )

                # Save results
                if write_consolidation_results:
                    write_parquet(
                        preds_df,
                        f"{AL_directory_path}/{consol_filename}_consol.parquet",
                    )
                    write_parquet(
                        preds_df_allRows,
                        f"{AL_directory_path}/{consol_filename}_full.parquet",
                    )
                    if write_csv:
                        preds_df.to_csv(
                            f"{AL_directory_path}/{consol_filename}_consol.csv",
                            index=False,
                        )
                        preds_df_allRows.to_csv(
                            f"{AL_directory_path}/{consol_filename}_full.csv",
                            index=False,
                        )

        # Define non-variable class as 1 - variable
        include_nonvar = False
        if f"vnv_{algorithm}" in preds_df.columns:
            include_nonvar = True
            preds_df[f"nonvar_{algorithm}"] = np.round(
                1 - preds_df[f"vnv_{algorithm}"], 2
            )

        if exclude_training_sources:
            # Get training set from config file
            training_set_config = self.config["training"]["dataset"]
            training_set_path = str(base_path / training_set_config)

            if training_set_path.endswith(".parquet"):
                training_set = read_parquet(training_set_path)
            elif training_set_path.endswith(".h5"):
                training_set = read_hdf(training_set_path)
            elif training_set_path.endswith(".csv"):
                training_set = pd.read_csv(training_set_path)
            else:
                raise ValueError(
                    "Training set must be in .parquet, .h5 or .csv format."
                )

            intersec = set.intersection(
                set(preds_df["obj_id"].values), set(training_set["obj_id"].values)
            )
            print(f"Dropping {len(intersec)} sources already in training set...")
            preds_df = preds_df.set_index("obj_id").drop(list(intersec)).reset_index()

        # Use trained model names to establish classes to train
        gen = os.walk(base_path / f"models_{algorithm}" / group)
        model_tags = [tag[1] for tag in gen]
        model_tags = model_tags[0]
        model_tags = np.array(model_tags)
        if include_nonvar:
            model_tags = np.concatenate([model_tags, ["nonvar"]])

        print(f"Selecting AL sample for {len(model_tags)} classes...")

        toPost_df = pd.DataFrame(columns=preds_df.columns)
        completed_dict = {}
        preds_df.set_index("obj_id", inplace=True)
        toPost_df.set_index("obj_id", inplace=True)

        # Fix random state to allow reproducible results
        rng = np.random.RandomState(9)

        # Reset min_class_examples if doAllSources is set
        if doAllSources:
            min_class_examples = len(preds_df)
            print(f"Selecting sample from all sources ({min_class_examples})")

        if not select_top_n:
            for tag in model_tags:
                # Idenfity all sources above probability threshold
                highprob_preds = preds_df[
                    preds_df[f"{tag}_{algorithm}"].values >= probability_threshold
                ]
                # Find existing sources in AL sample above probability threshold
                existing_df = toPost_df[
                    toPost_df[f"{tag}_{algorithm}"].values >= probability_threshold
                ]
                existing_count = len(existing_df)

                # Determine number of sources needed to reach at least min_class_examples
                still_to_post_count = min_class_examples - existing_count

                if still_to_post_count > 0:
                    if len(highprob_preds) >= still_to_post_count:
                        # Randomly select from remaining examples without replacement
                        highprob_toPost = rng.choice(
                            highprob_preds.drop(existing_df.index).index,
                            still_to_post_count,
                            replace=False,
                        )
                        concat_toPost_df = highprob_preds.loc[highprob_toPost]
                    else:
                        # If sample is limited, use all remaining available examples
                        concat_toPost_df = highprob_preds

                toPost_df = pd.concat([toPost_df, concat_toPost_df], axis=0)
                toPost_df.drop_duplicates(keep="first", inplace=True)

        else:
            # Select top N classifications above probability threshold for all classes
            print(
                f"Selecting top {min_class_examples} classifications above P = {probability_threshold}..."
            )

            preds_df.reset_index(inplace=True)
            topN_df = pd.DataFrame()
            class_list = [f"{t}_{algorithm}" for t in model_tags]

            for tag in model_tags:
                goodprob_preds = preds_df[
                    preds_df[f"{tag}_{algorithm}"].values >= probability_threshold
                ]

                if not include_all_highprob_labels:
                    # Return only the top N probabilities for each class, even if other high-probability classifications are excluded
                    topN_preds = (
                        goodprob_preds[
                            [
                                "obj_id",
                                "survey_id",
                                "ra",
                                "dec",
                                "period",
                                f"{tag}_{algorithm}",
                            ]
                        ]
                        .sort_values(by=f"{tag}_{algorithm}", ascending=False)
                        .iloc[:min_class_examples]
                        .reset_index(drop=True)
                    )

                else:
                    # Include not only the top N probabilities for each class but also any other classifications above probability_threshold for these sources
                    topN_preds = (
                        goodprob_preds.sort_values(
                            by=f"{tag}_{algorithm}", ascending=False
                        )
                        .iloc[:min_class_examples]
                        .reset_index(drop=True)
                    )

                    topN_preds = topN_preds.apply(
                        self._drop_low_probs,
                        class_list=class_list,
                        threshold=probability_threshold,
                    )

                topN_df = pd.concat([topN_df, topN_preds]).reset_index(drop=True)

            toPost_df = topN_df.fillna(0.0).groupby("obj_id").max().reset_index()

        for tag in model_tags:
            # Make metadata dictionary of example count per class
            completed_dict[f"{tag}_{algorithm}"] = int(
                np.sum(toPost_df[f"{tag}_{algorithm}"].values >= probability_threshold)
            )

        final_toPost = toPost_df.reset_index(drop=True)

        if not doNotSave:
            # Write parquet and csv files
            write_parquet(final_toPost, f"{AL_directory_path}/{al_filename}.parquet")
            if write_csv:
                final_toPost.to_csv(
                    f"{AL_directory_path}/{al_filename}.csv", index=False
                )

            # Write metadata
            meta_filepath = f"{AL_directory_path}/meta.json"
            with open(meta_filepath, "w") as f:
                try:
                    json.dump(completed_dict, f)  # dump dictionary to a json file
                except Exception as e:
                    print("error dumping to json, message: ", e)

        yield final_toPost

    def test_limited(self):
        """
        Test workflows that do not require a kowalski connection

        :return:
        """
        import uuid

        # create a mock dataset and check that the training pipeline works
        dataset = f"{uuid.uuid4().hex}_orig.csv"
        path_mock = self.base_path / "data" / "training"
        group_mock = "scope_test_limited"

        try:
            with status("Test training"):
                print()

                period_suffix_config = self.config["features"]["info"]["period_suffix"]

                if not path_mock.exists():
                    path_mock.mkdir(parents=True, exist_ok=True)

                all_feature_names = self.config["features"]["ontological"]
                feature_names_orig = [
                    key
                    for key in all_feature_names
                    if forgiving_true(all_feature_names[key]["include"])
                ]

                feature_names = feature_names_orig.copy()
                if not (
                    (period_suffix_config is None) | (period_suffix_config == "None")
                ):
                    periodic_bool = [
                        all_feature_names[x]["periodic"] for x in feature_names
                    ]
                    for j, name in enumerate(feature_names):
                        if periodic_bool[j]:
                            feature_names[j] = f"{name}_{period_suffix_config}"

                class_names = [
                    self.config["training"]["classes"][class_name]["label"]
                    for class_name in self.config["training"]["classes"]
                ]

                entries = []
                for i in range(1000):
                    entry = {
                        **{
                            feature_name: np.random.normal(0, 0.1)
                            for feature_name in feature_names
                        },
                        **{
                            class_name: np.random.choice([0, 1])
                            for class_name in class_names
                        },
                        **{"non-variable": np.random.choice([0, 1])},
                        **{"dmdt": np.abs(np.random.random((26, 26))).tolist()},
                    }
                    entries.append(entry)

                df_mock_orig = pd.DataFrame.from_records(entries)
                df_mock_orig.to_csv(path_mock / dataset, index=False)

                algorithms = ["xgb"]
                try:
                    import tensorflow  # noqa: F401

                    algorithms.append("dnn")
                except ImportError:
                    print("tensorflow not installed, skipping DNN training test")

                model_paths = []

                # Train twice: once on Kowalski features, once on generated features with different periodic feature names
                for algorithm in algorithms:
                    tag = "vnv"
                    if algorithm == "xgb":
                        extension = "json"
                    elif algorithm == "dnn":
                        extension = "h5"
                    time_tag = self.train(
                        tag=tag,
                        path_dataset=path_mock / dataset,
                        batch_size=32,
                        epochs=3,
                        verbose=True,
                        save=True,
                        evaluate_test=False,
                        algorithm=algorithm,
                        skip_cv=True,
                        group=group_mock,
                    )
                    path_model = (
                        self.base_path
                        / f"models_{algorithm}"
                        / group_mock
                        / tag
                        / f"{tag}.{time_tag}.{extension}"
                    )
                    model_paths += [path_model]

            print("model_paths", model_paths)

        finally:
            # clean up after thyself
            (path_mock / dataset).unlink()

            # Remove trained model artifacts, but keep models_xgb and models_dnn directories
            for path in model_paths:
                shutil.rmtree(path.parent.parent)

    def parse_run_test(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--doGPU",
            action="store_true",
            help="if set, use GPU-accelerated period algorithm",
        )
        args, _ = parser.parse_known_args()
        self.test(**vars(args))

    def test(self, doGPU=False):
        """
        Test different workflows

        :return:
        """
        import uuid
        from tools import (
            generate_features,
            get_quad_ids,
            get_features,
            inference,
            combine_preds,
        )
        from .surveys.fritz import get_lightcurves_via_coords

        # Test feature generation
        with status("Test generate_features"):
            print()
            test_field, test_ccd, test_quad = 297, 2, 2
            test_feature_directory = "generated_features"
            test_feature_filename = "testFeatures"
            n_sources = 3

            _ = generate_features.generate_features(
                doCPU=not doGPU,
                doGPU=doGPU,
                field=test_field,
                ccd=test_ccd,
                quad=test_quad,
                dirname=test_feature_directory,
                filename=test_feature_filename,
                limit=n_sources,
                doCesium=True,
                stop_early=True,
                Ncore=1,
                min_n_lc_points=50,
                doRemoveTerrestrial=True,
                doScaleMinPeriod=True,
            )

            path_to_features = self.config.get("feature_generation").get(
                "path_to_features"
            )
            if path_to_features is None:
                path_gen_features = (
                    self.base_path
                    / test_feature_directory
                    / f"field_{test_field}"
                    / f"{test_feature_filename}_field_{test_field}_ccd_{test_ccd}_quad_{test_quad}.parquet"
                )
            else:
                path_gen_features = (
                    pathlib.Path(path_to_features)
                    / test_feature_directory
                    / f"field_{test_field}"
                    / f"{test_feature_filename}_field_{test_field}_ccd_{test_ccd}_quad_{test_quad}.parquet"
                )

        with status("Test get_lightcurves_via_coords"):
            print()
            _ = get_lightcurves_via_coords(
                kowalski_instances=self.kowalski, ra=40.0, dec=50.0, radius=2.0
            )

        with status("Test get_cone_ids"):
            print()
            _ = get_quad_ids.get_cone_ids(
                obj_id_list=["obj1", "obj2", "obj3"],
                ra_list=[40.0, 41.0, 42.0],
                dec_list=[50.0, 51.0, 52.0],
            )

        src_catalog = self.config["kowalski"]["collections"]["sources"]
        with status("Test get_ids_loop and get_field_ids"):
            print()
            _, lst = get_quad_ids.get_ids_loop(
                get_quad_ids.get_field_ids,
                catalog=src_catalog,
                field=297,
                ccd_range=3,
                quad_range=4,
                limit=10,
                stop_early=True,
                save=False,
            )

        with status("Test get_features_loop and get_features"):
            print()
            test_ftrs, outfile = get_features.get_features_loop(
                get_features.get_features,
                source_ids=lst[0],
                features_catalog=self.config["kowalski"]["collections"]["features"],
                field=297,
                limit_per_query=5,
                max_sources=10,
                save=False,
            )

            if path_to_features is None:
                testpath = pathlib.Path(outfile)
                testpath = testpath.parent.parent
            else:
                testpath = pathlib.Path(path_to_features) / "features"
            # Use 'field_0' as test directory to avoid removing any existing data locally
            testpath_features = testpath / "field_0"

            if not testpath_features.exists():
                testpath_features.mkdir(parents=True, exist_ok=True)
            write_parquet(test_ftrs, str(testpath_features / "field_0_iter_0.parquet"))

        # create a mock dataset and check that the training pipeline works
        dataset_orig = f"{uuid.uuid4().hex}_orig.csv"
        dataset = f"{uuid.uuid4().hex}.csv"
        path_mock = self.base_path / "data" / "training"
        group_mock = "scope_test"

        try:
            with status("Test training"):
                print()

                period_suffix_config = (
                    self.config.get("features").get("info").get("period_suffix")
                )
                if doGPU:
                    if period_suffix_config not in [
                        "ELS",
                        "ECE",
                        "EAOV",
                        "ELS_ECE_EAOV",
                    ]:
                        period_suffix_test = "ELS_ECE_EAOV"
                    else:
                        period_suffix_test = period_suffix_config
                else:
                    if period_suffix_config not in ["LS", "CE", "AOV", "LS_CE_AOV"]:
                        period_suffix_test = "LS"
                    else:
                        period_suffix_test = period_suffix_config

                if not path_mock.exists():
                    path_mock.mkdir(parents=True, exist_ok=True)

                all_feature_names = self.config["features"]["ontological"]
                feature_names_orig = [
                    key
                    for key in all_feature_names
                    if forgiving_true(all_feature_names[key]["include"])
                ]

                feature_names_new = feature_names_orig.copy()
                if not (
                    (period_suffix_config is None) | (period_suffix_config == "None")
                ):
                    periodic_bool = [
                        all_feature_names[x]["periodic"] for x in feature_names_new
                    ]
                    for j, name in enumerate(feature_names_new):
                        if periodic_bool[j]:
                            feature_names_new[j] = f"{name}_{period_suffix_config}"

                feature_names = feature_names_orig.copy()
                if not ((period_suffix_test is None) | (period_suffix_test == "None")):
                    periodic_bool = [
                        all_feature_names[x]["periodic"] for x in feature_names
                    ]
                    for j, name in enumerate(feature_names):
                        if periodic_bool[j]:
                            feature_names[j] = f"{name}_{period_suffix_test}"

                class_names = [
                    self.config["training"]["classes"][class_name]["label"]
                    for class_name in self.config["training"]["classes"]
                ]

                entries = []
                for i in range(1000):
                    entry = {
                        **{
                            feature_name: np.random.normal(0, 0.1)
                            for feature_name in feature_names_new
                        },
                        **{
                            class_name: np.random.choice([0, 1])
                            for class_name in class_names
                        },
                        **{"non-variable": np.random.choice([0, 1])},
                        **{"dmdt": np.abs(np.random.random((26, 26))).tolist()},
                    }
                    entries.append(entry)

                df_mock_orig = pd.DataFrame.from_records(entries)
                df_mock_orig.to_csv(path_mock / dataset_orig, index=False)

                entries = []
                for i in range(1000):
                    entry = {
                        **{
                            feature_name: np.random.normal(0, 0.1)
                            for feature_name in feature_names
                        },
                        **{
                            class_name: np.random.choice([0, 1])
                            for class_name in class_names
                        },
                        **{"non-variable": np.random.choice([0, 1])},
                        **{"dmdt": np.abs(np.random.random((26, 26))).tolist()},
                    }
                    entries.append(entry)

                df_mock = pd.DataFrame.from_records(entries)
                df_mock.to_csv(path_mock / dataset, index=False)

                algorithms = ["xgb", "dnn"]
                model_paths_orig = []

                # Train twice: once on Kowalski features, once on generated features with different periodic feature names
                for algorithm in algorithms:
                    tag = "vnv"
                    if algorithm == "xgb":
                        extension = "json"
                    elif algorithm == "dnn":
                        extension = "h5"
                    time_tag = self.train(
                        tag=tag,
                        path_dataset=path_mock / dataset_orig,
                        batch_size=32,
                        epochs=3,
                        verbose=True,
                        save=True,
                        evaluate_test=False,
                        algorithm=algorithm,
                        skip_cv=True,
                        group=group_mock,
                    )
                    path_model = (
                        self.base_path
                        / f"models_{algorithm}"
                        / group_mock
                        / tag
                        / f"{tag}.{time_tag}.{extension}"
                    )
                    model_paths_orig += [path_model]

                model_paths = []
                for algorithm in algorithms:
                    tag = "vnv"
                    if algorithm == "xgb":
                        extension = "json"
                    elif algorithm == "dnn":
                        extension = "h5"
                    time_tag = self.train(
                        tag=tag,
                        path_dataset=path_mock / dataset,
                        batch_size=32,
                        epochs=3,
                        verbose=True,
                        save=True,
                        evaluate_test=False,
                        algorithm=algorithm,
                        skip_cv=True,
                        period_suffix=period_suffix_test,
                        group=group_mock,
                    )
                    path_model = (
                        self.base_path
                        / f"models_{algorithm}"
                        / group_mock
                        / tag
                        / f"{tag}.{time_tag}.{extension}"
                    )
                    model_paths += [path_model]

            print("model_paths_orig", model_paths_orig)
            print("model_paths", model_paths)

            with status("Test inference (queried features)"):
                print()
                print(model_paths[1])
                _, preds_filename_dnn_orig = inference.run_inference(
                    paths_models=[str(model_paths_orig[1])],
                    model_class_names=[tag],
                    field=0,
                    whole_field=True,
                    trainingSet=df_mock_orig,
                )
                print(model_paths[0])
                _, preds_filename_xgb_orig = inference.run_inference(
                    paths_models=[str(model_paths_orig[0])],
                    model_class_names=[tag],
                    field=0,
                    whole_field=True,
                    trainingSet=df_mock_orig,
                    xgb_model=True,
                )

            with status("Test inference (generated features)"):
                print()
                _, preds_filename_dnn = inference.run_inference(
                    paths_models=[str(model_paths[1])],
                    model_class_names=[tag],
                    field=test_field,
                    ccd=test_ccd,
                    quad=test_quad,
                    trainingSet=df_mock,
                    feature_directory=test_feature_directory,
                    feature_file_prefix=test_feature_filename,
                    period_suffix=period_suffix_test,
                    no_write_metadata=True,
                )
                print()
                _, preds_filename_xgb = inference.run_inference(
                    paths_models=[str(model_paths[0])],
                    model_class_names=[tag],
                    field=test_field,
                    ccd=test_ccd,
                    quad=test_quad,
                    trainingSet=df_mock,
                    xgb_model=True,
                    feature_directory=test_feature_directory,
                    feature_file_prefix=test_feature_filename,
                    period_suffix=period_suffix_test,
                    no_write_metadata=True,
                )

            with status("Test combine_preds"):
                combine_preds.combine_preds(specific_field=0, save=False)

            with status("Test select_fritz_sample"):
                print()
                _ = self.select_fritz_sample(
                    [0],
                    probability_threshold=0.0,
                    doNotSave=True,
                )
                _ = self.select_fritz_sample(
                    [0],
                    select_top_n=True,
                    include_all_highprob_labels=True,
                    min_class_examples=3,
                    probability_threshold=0.0,
                    doNotSave=True,
                )
                _ = self.select_fritz_sample(
                    [0],
                    probability_threshold=0.0,
                    doNotSave=True,
                    algorithm="xgb",
                )
                _ = self.select_fritz_sample(
                    [0],
                    select_top_n=True,
                    include_all_highprob_labels=True,
                    min_class_examples=3,
                    probability_threshold=0.0,
                    doNotSave=True,
                    algorithm="xgb",
                )

        finally:
            # clean up after thyself
            (path_mock / dataset_orig).unlink()
            (path_mock / dataset).unlink()
            os.remove(path_gen_features)
            (testpath_features / "field_0_iter_0.parquet").unlink()
            os.rmdir(testpath_features)
            (preds_filename_dnn_orig).unlink()
            (preds_filename_xgb_orig).unlink()
            (preds_filename_dnn).unlink()
            (preds_filename_xgb).unlink()
            (preds_filename_dnn_orig.parent / "meta.json").unlink()
            (preds_filename_xgb_orig.parent / "meta.json").unlink()
            os.rmdir(preds_filename_dnn_orig.parent)
            os.rmdir(preds_filename_xgb_orig.parent)

            # Remove trained model artifacts, but keep models_xgb and models_dnn directories
            for path in model_paths:
                shutil.rmtree(path.parent.parent)
