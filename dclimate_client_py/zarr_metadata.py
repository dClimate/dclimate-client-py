from dclimate_client_py.dclimate_zarr_errors import (
    BucketNotFoundError,
    PathNotFoundError,
    ZarrClientError,
)
from dclimate_client_py.s3_retrieval import get_s3_fs
import typing
import os
import json


def get_standard_collections(bucket_name: str) -> typing.List[str]:
    catalog_metadata = get_catalog_metadata(bucket_name)
    return [
        os.path.splitext(collection["href"])[0].split("collections/")[1]
        for collection in catalog_metadata["links"]
        if "forecast" not in collection["title"].lower()
        and "root catalog" not in collection["title"]
    ]


def get_forecast_collections(bucket_name: str) -> typing.List[str]:
    catalog_metadata = get_catalog_metadata(bucket_name)
    return [
        os.path.splitext(collection["href"])[0].split("collections/")[1]
        for collection in catalog_metadata["links"]
        if "forecast" in collection["title"].lower()
        and "root catalog" not in collection["title"]
    ]


def get_collection_metadata(bucket_name: str, collection_name: str):
    s3 = get_s3_fs()
    _validate_bucket_name(bucket_name)
    collection_metadata_path = (
        f"{bucket_name}/metadata/collections/{collection_name}.json"
    )
    _validate_path(collection_metadata_path)
    collection_metadata = s3.cat_file(collection_metadata_path)
    return json.loads(collection_metadata)


def get_collection_datasets(bucket_name: str, collection_name: str):
    s3 = get_s3_fs()
    _validate_bucket_name(bucket_name)
    collection_file_path = f"{bucket_name}/metadata/collections/{collection_name}.json"
    _validate_path(collection_file_path)
    collection_file_content = s3.cat_file(collection_file_path)
    try:
        collection_file_content_as_dict = json.loads(collection_file_content)
        links = collection_file_content_as_dict.get("links") or []
        return [
            _extract_file_name_from_path(link.get("href"))
            for link in links
            if link.get("rel") == "item" and link.get("href")
        ]

    except ValueError:
        raise ZarrClientError(
            f"There is an error reading the file: {collection_file_path}"
        )


def get_dataset_metadata(bucket_name: str, dataset_name: str):
    s3 = get_s3_fs()
    _validate_bucket_name(bucket_name)
    dataset_metadata_file_path = f"{bucket_name}/metadata/datasets/{dataset_name}.json"
    _validate_path(dataset_metadata_file_path)
    dataset_metadata = s3.cat_file(dataset_metadata_file_path)
    return json.loads(dataset_metadata)


def get_catalog_metadata(bucket_name: str):
    s3 = get_s3_fs()
    _validate_bucket_name(bucket_name)
    metadata_path = f"{bucket_name}/metadata"
    _validate_path(metadata_path)
    metadata_folder_content = s3.ls(metadata_path, detail=False)
    data_catalog_files = [
        item
        for item in metadata_folder_content
        if item.lower().endswith(".json") and "Data Catalog" in item
    ]
    if len(data_catalog_files) > 1:
        raise ZarrClientError("There is more than one Data Catalog object")
    if len(data_catalog_files) == 0:
        raise ZarrClientError("There is not any Data Catalog object")
    catalog_metadata = s3.cat_file(data_catalog_files[0])
    return json.loads(catalog_metadata)


def _validate_bucket_name(bucket_name: str):
    try:
        _validate_path(bucket_name)
    except PathNotFoundError:
        raise BucketNotFoundError(f"Bucket {bucket_name} does not exist")


def _validate_path(path: str):
    s3 = get_s3_fs()
    exists = s3.exists(f"{path}")
    if not exists:
        raise PathNotFoundError(f"Path {path} does not exist")


def _extract_file_name_from_path(path: str):
    return os.path.splitext(os.path.basename(path))[0]
