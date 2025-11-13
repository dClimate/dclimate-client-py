import pytest

import dclimate_client_py.s3_retrieval as s3_retrieval
from dclimate_client_py.dclimate_zarr_errors import DatasetNotFoundError
import json
import os
from collections import namedtuple


class TestS3Retrieval:
    @pytest.fixture
    def fake_s3fs(self, mocker):
        fake_s3fs = mocker.Mock()
        mocker.patch(
            "dclimate_client_py.s3_retrieval.get_s3_fs", return_value=fake_s3fs
        )
        return fake_s3fs

    class TestGetDataSetFromS3Function:
        def test__given_a_dataset_name_and_bucket_name__it_fetch_the_dataset(
            self, mocker, fake_s3fs
        ):
            dataset_name = "copernicus_ocean_salinity_1p5_meters-daily"
            bucket_name = "zarr-prod"
            s3Map_mock = mocker.patch("dclimate_client_py.s3_retrieval.S3Map")

            mock_dataset = namedtuple("Dataset", ["update_in_progress"])(False)
            mocker.patch("xarray.open_zarr", return_value=mock_dataset)

            ds = s3_retrieval.get_dataset_from_s3(dataset_name, bucket_name)

            assert ds is mock_dataset
            s3Map_mock.assert_called_with(
                f"s3://{bucket_name}/datasets/{dataset_name}.zarr",
                s3=fake_s3fs,
            )

        def test__given_a_dataset_with_initial_parse_true__it_raises_error(
            self, mocker
        ):
            mock_dataset = namedtuple(
                "Dataset", ["update_in_progress", "initial_parse"]
            )(True, True)
            mocker.patch("xarray.open_zarr", return_value=mock_dataset)

            with pytest.raises(
                DatasetNotFoundError,
                match="Dataset fake-dataset is undergoing initial parse, retry request later",
            ):
                s3_retrieval.get_dataset_from_s3("fake-dataset", "zarr-prod")

    class TestListS3DatasetsFunction:
        def test__given_a_bucket_name__then__returns_all_zarr_datasets(
            self, mocker, fake_s3fs
        ):
            bucket_name = "zarr-prod"
            datasets = [
                "zarr-dev/datasets/chirps_final_05-daily.zarr",
                "zarr-dev/datasets/chirps_prelim_05-daily.zarr",
                "zarr-dev/datasets/copernicus_ocean_salinity_1p5_meters-daily.zarr",
                "zarr-dev/datasets/cpc_precip_global-daily.zarr",
                "zarr-dev/datasets/cpc_precip_us-daily.zarr",
                "zarr-dev/datasets/cpc_temp_max-daily.zarr",
                "zarr-dev/datasets/cpc_temp_min-daily.zarr",
                "zarr-dev/datasets/era5_precip-hourly.zarr",
            ]
            fake_s3fs.ls = mocker.Mock(return_value=datasets)

            s3_datasets = s3_retrieval.list_s3_datasets(bucket_name)

            assert s3_datasets == [
                "chirps_final_05-daily",
                "chirps_prelim_05-daily",
                "copernicus_ocean_salinity_1p5_meters-daily",
                "cpc_precip_global-daily",
                "cpc_precip_us-daily",
                "cpc_temp_max-daily",
                "cpc_temp_min-daily",
                "era5_precip-hourly",
            ]

    class TestGetMetadataByS3Key:
        def test__given_key_and_bucket_name__then__returns_metadata(
            self, mocker, fake_s3fs
        ):
            bucket_name = "zarr-prod"
            key = "chirps_final_05-daily"
            zarr_metadata = json.dumps(
                {
                    "CDI": "Climate Data Interface version 2.0.4 (https://mpimet.mpg.de/cdi)",
                    "CDO": "Climate Data Operators version 2.0.4 (https://mpimet.mpg.de/cdo)",
                    "bbox": [-179.975, -49.975, 179.975, 49.975],
                }
            ).encode()
            fake_s3fs.cat = mocker.Mock(return_value=zarr_metadata)

            metadata = s3_retrieval.get_metadata_by_s3_key(key, bucket_name)

            assert metadata == json.loads(zarr_metadata)


def test_identical_s3fs_instance_used_when_profile_set():
    os.environ["ZARR_AWS_PROFILE_NAME"] = "test"
    s3fs = s3_retrieval.get_s3_fs()
    s3fs2 = s3_retrieval.get_s3_fs()
    assert s3fs == s3fs2
