# flake8: noqa: E501  line too long
import json
import pathlib

import pytest

from dclimate_client_py.dclimate_zarr_errors import (
    BucketNotFoundError,
    PathNotFoundError,
    ZarrClientError,
)

from dclimate_client_py.zarr_metadata import (
    get_standard_collections,
    get_forecast_collections,
    get_collection_metadata,
    get_collection_datasets,
    get_dataset_metadata,
    get_catalog_metadata,
)

HERE = pathlib.Path(__file__).parent
METADATA = HERE / "data" / "zarr-dev" / "metadata"


class TestZarrMetadata:
    @pytest.fixture()
    def s3_fs(self, mocker):
        s3fs_mock = mocker.Mock()
        mocker.patch(
            "dclimate_client_py.zarr_metadata.get_s3_fs", return_value=s3fs_mock
        )
        return s3fs_mock

    class TestGetCollectionsFunction:
        def test__given_an_invalid_bucket_name__then__an_error_is_thrown(
            self, mocker, s3_fs
        ):
            s3_fs.exists = mocker.Mock(return_value=False)
            invalid_bucket_name = "invalid_bucket_name"

            with pytest.raises(BucketNotFoundError) as e:
                get_standard_collections(invalid_bucket_name)

            assert str(e.value) == f"Bucket {invalid_bucket_name} does not exist"

        def test__given_a_valid_bucket_name_and_bucket_contains_collections__then__they_are_returned(
            self, mocker
        ):
            mocker.patch(
                "dclimate_client_py.zarr_metadata.get_catalog_metadata",
                return_value=json.loads(open(METADATA / "Data Catalog.json").read()),
            )
            bucket_name = "zarr-dev"

            collections = get_standard_collections(bucket_name)
            assert collections == ["CPC", "ERA5"]

        def test__given_a_valid_bucket_name_and_bucket_contains_forecast_collections__then__they_are_returned(
            self, mocker
        ):
            mocker.patch(
                "dclimate_client_py.zarr_metadata.get_catalog_metadata",
                return_value=json.loads(open(METADATA / "Data Catalog.json").read()),
            )
            bucket_name = "zarr-dev"

            collections = get_forecast_collections(bucket_name)
            assert collections == ["GFS"]

        def test__given_a_valid_bucket_name_and_bucket_does_not_contain_collections__then__empty_array_is_returned(
            self, mocker, s3_fs
        ):
            mocker.patch(
                "dclimate_client_py.zarr_metadata.get_catalog_metadata",
                return_value=json.loads(
                    open(METADATA / "Empty Data Catalog.json").read()
                ),
            )
            bucket_name = "zarr-dev"

            collections = get_standard_collections(bucket_name)

            assert collections == []

    class TestGetCollectionMetadata:
        def test_given_an_invalid_bucket_name_or_collection_name__then__an_error_is_thrown(
            self, mocker, s3_fs
        ):
            s3_fs.exists = mocker.Mock(return_value=False)
            invalid_bucket_name = "invalid_bucket_name"

            with pytest.raises(BucketNotFoundError) as e:
                get_catalog_metadata(invalid_bucket_name)

            assert str(e.value) == f"Bucket {invalid_bucket_name} does not exist"

        def test_given_a_valid_bucket_name_and_collection_name__then__metadata_content_file_is_returned(
            self, mocker, s3_fs
        ):
            fake_collection_metadata = json.dumps({"key": "value"}).encode()
            s3_fs.cat_file = mocker.Mock(return_value=fake_collection_metadata)
            bucket_name = "zarr-dev"
            collection_name = "CHIRPS"

            result = get_collection_metadata(bucket_name, collection_name)

            assert result == json.loads(fake_collection_metadata)

    class TestGetCollectionDatasetsFunction:
        def test_given_an_invalid_bucket_name__then__an_error_is_thrown(
            self, mocker, s3_fs
        ):
            s3_fs.exists = mocker.Mock(return_value=False)
            invalid_bucket_name = "invalid_bucket_name"
            collection_name = "CHIRPS"
            with pytest.raises(BucketNotFoundError) as e:
                get_collection_datasets(invalid_bucket_name, collection_name)

            assert str(e.value) == f"Bucket {invalid_bucket_name} does not exist"

        def test_given_an_invalid_collection_name__then__an_error_is_thrown(
            self, mocker, s3_fs
        ):
            s3_fs.exists = mocker.Mock(side_effect=[True, False])
            bucket_name = "zarr-dev"
            invalid_collection_name = "invalid"
            with pytest.raises(PathNotFoundError) as e:
                get_collection_datasets(bucket_name, invalid_collection_name)

            assert (
                str(e.value)
                == f"Path {bucket_name}/metadata/collections/{invalid_collection_name}.json does not exist"
            )

        def test_given_valid_bucket_and_collection_names__then__datasets_are_returned(
            self, mocker, s3_fs
        ):
            with open(METADATA / "collections/CHIRPS.json", "rb") as file:
                s3_fs.cat_file = mocker.Mock(return_value=file.read())
            bucket_name = "zarr-dev"
            collection_name = "CHIRPS"

            result = get_collection_datasets(bucket_name, collection_name)

            assert result == ["chirps_final_05-daily", "chirps_final_05-monthly"]

        def test_given_valid_bucket_and_collection_names__and_collection_file_does_not_have_links_then__empty_result(
            self, mocker, s3_fs
        ):
            s3_fs.cat_file = mocker.Mock(return_value=b'{"other_key": []}')
            bucket_name = "zarr-dev"
            collection_name = "CHIRPS"

            result = get_collection_datasets(bucket_name, collection_name)

            assert result == []

        def test_given_valid_bucket_and_collection_names__and_collection_file_does_not_have_links_to_datasets_then__empty_result(
            self, mocker, s3_fs
        ):
            s3_fs.cat_file = mocker.Mock(
                return_value=b'{"links": [ {"rel": "root", "href": "s3://zarr-dev/any.json","type": "application/json", "title": "random"}]}'
            )
            bucket_name = "zarr-dev"
            collection_name = "CHIRPS"

            result = get_collection_datasets(bucket_name, collection_name)

            assert result == []

        def test_given_valid_bucket_and_collection_names__and_collection_file_contains_links_then__only_valid_datasets_are_returned(
            self, mocker, s3_fs
        ):
            s3_fs.cat_file = mocker.Mock(
                return_value=b'{"links": [ '
                b'{"rel": "item", "href": "s3://zarr-dev/a.json","type": "application/json", "title": "random"},'
                b'{"rel": "item", "type": "application/json", "title": "random"}'
                b"]}"
            )
            bucket_name = "zarr-dev"
            collection_name = "CHIRPS"

            result = get_collection_datasets(bucket_name, collection_name)

            assert result == ["a"]

        def test_given_valid_bucket_and_collection_names__and_collection_file_is_not_a_valid_json_then__error_is_thrown(
            self, mocker, s3_fs
        ):
            s3_fs.cat_file = mocker.Mock(return_value=b"Invalid_json_content")
            bucket_name = "zarr-dev"
            collection_name = "CHIRPS"
            with pytest.raises(ZarrClientError) as e:
                get_collection_datasets(bucket_name, collection_name)

            assert (
                str(e.value)
                == f"There is an error reading the file: {bucket_name}/metadata/collections/{collection_name}.json"
            )

    class TestGetDatasetMetadata:
        def test_given_an_invalid_bucket_name__then__an_error_is_thrown(
            self, mocker, s3_fs
        ):
            s3_fs.exists = mocker.Mock(return_value=False)
            invalid_bucket_name = "invalid_bucket_name"
            dataset_name = "chirps_final_05-daily"

            with pytest.raises(BucketNotFoundError) as e:
                get_dataset_metadata(invalid_bucket_name, dataset_name)

            assert str(e.value) == f"Bucket {invalid_bucket_name} does not exist"

        def test_given_an_invalid_dataset_name__then__an_error_is_thrown(
            self, mocker, s3_fs
        ):
            s3_fs.exists = mocker.Mock(side_effect=[True, False])
            bucket_name = "zarr-dev"
            invalid_dataset_name = "invalid"

            with pytest.raises(PathNotFoundError) as e:
                get_dataset_metadata(bucket_name, invalid_dataset_name)

            assert (
                str(e.value)
                == f"Path {bucket_name}/metadata/datasets/{invalid_dataset_name}.json does not exist"
            )

        def test_given_a_valid_bucket_name_and_dataset_name__then__metadata_content_file_is_returned(
            self, mocker, s3_fs
        ):
            fake_dataset_metadata = json.dumps({"key": "value"}).encode()
            s3_fs.cat_file = mocker.Mock(return_value=fake_dataset_metadata)
            bucket_name = "zarr-dev"
            dataset_name = "chirps_final_05-daily"

            result = get_dataset_metadata(bucket_name, dataset_name)

            assert result == json.loads(fake_dataset_metadata)

    class TestGetCatalogMetadata:
        def test_given_an_invalid_bucket_name__then__an_error_is_thrown(
            self, mocker, s3_fs
        ):
            s3_fs.exists = mocker.Mock(return_value=False)
            invalid_bucket_name = "invalid_bucket_name"

            with pytest.raises(BucketNotFoundError) as e:
                get_catalog_metadata(invalid_bucket_name)

            assert str(e.value) == f"Bucket {invalid_bucket_name} does not exist"

        def test_if_there_is_not_metadata_folder__then__an_error_is_thrown(
            self, mocker, s3_fs
        ):
            s3_fs.exists = mocker.Mock(side_effect=[True, False])
            bucket_name = "zarr-dev"
            # invalid_dataset_name = "invalid"

            with pytest.raises(PathNotFoundError) as e:
                get_catalog_metadata(bucket_name)

            assert str(e.value) == f"Path {bucket_name}/metadata does not exist"

        def test_if_there_is_more_than_one_data_catalog_object__then__an_error_is_thrown(
            self, mocker, s3_fs
        ):
            fake_metadata_folder_content = [
                "zarr-dev/metadata/My Organization Data Catalog.json",
                "zarr-dev/metadata/dClimate Data Catalog - test.json",
                "zarr-dev/metadata/collections",
                "zarr-dev/metadata/datasets",
            ]

            s3_fs.ls = mocker.Mock(return_value=fake_metadata_folder_content)
            bucket_name = "zarr-dev"
            with pytest.raises(ZarrClientError) as e:
                get_catalog_metadata(bucket_name)

            assert str(e.value) == "There is more than one Data Catalog object"

        def test__if_there_is_no_data_catalog_object__then__an_error_is_thrown(
            self, mocker, s3_fs
        ):
            fake_metadata_folder_content = [
                "zarr-dev/metadata/collections",
                "zarr-dev/metadata/datasets",
            ]

            s3_fs.ls = mocker.Mock(return_value=fake_metadata_folder_content)
            bucket_name = "zarr-dev"
            with pytest.raises(ZarrClientError) as e:
                get_catalog_metadata(bucket_name)

            assert str(e.value) == "There is not any Data Catalog object"

        def test__if_there_exactly_one_data_catalog_object_then__its_content_is_returned(
            self, mocker, s3_fs
        ):
            fake_metadata_folder_content = [
                "zarr-dev/metadata/My Organization Data Catalog.json",
                "zarr-dev/metadata/collections",
                "zarr-dev/metadata/datasets",
            ]

            s3_fs.ls = mocker.Mock(return_value=fake_metadata_folder_content)
            bucket_name = "zarr-dev"
            data_catalog_object_content = mocker.Mock()
            data_catalog_object_content_as_dict = mocker.Mock()
            s3_fs.cat_file = mocker.Mock(return_value=data_catalog_object_content)
            fake_json_loads = mocker.patch(
                "json.loads", return_value=data_catalog_object_content_as_dict
            )

            result = get_catalog_metadata(bucket_name)

            assert result == data_catalog_object_content_as_dict
            s3_fs.cat_file.assert_called_with(fake_metadata_folder_content[0])
            fake_json_loads.assert_called_with(data_catalog_object_content)
