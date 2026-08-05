import unittest
from unittest.mock import Mock

from dclimate_client_py.ceramic_api import (
    CitationInfo,
    DatasetVersion,
    DatasetVersionListing,
    build_gateway_url,
    filter_anchored_versions,
    get_citation,
    get_citation_from_url,
    get_exact_version,
    get_exact_version_from_url,
    get_latest_anchored_version,
    get_latest_metadata,
    list_versions,
    list_versions_from_url,
)


def _mock_response(payload):
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


class CeramicApiTests(unittest.TestCase):
    def test_list_versions_builds_filters_and_parses_response(self):
        session = Mock()
        session.get.return_value = _mock_response(
            {
                "dataset": "eagle-temp",
                "streamId": "stream-123",
                "versions": [
                    {
                        "dataset": "eagle-temp",
                        "cid": "cid-1",
                        "commitId": "commit-1",
                        "streamId": "stream-123",
                        "versionLabel": "2026-05-01",
                        "verification": {"anchorStatus": "anchored"},
                    },
                    {
                        "dataset": "eagle-temp",
                        "cid": "cid-2",
                        "commitId": "commit-2",
                        "streamId": "stream-123",
                        "versionLabel": "2026-05-02",
                        "verification": {"anchorStatus": "pending"},
                    },
                ],
            }
        )

        result = list_versions(
            "eagle-temp",
            anchored=True,
            is_citable=False,
            version_label="2026-05",
            session=session,
        )

        self.assertIsInstance(result, DatasetVersionListing)
        self.assertEqual(result.dataset, "eagle-temp")
        self.assertEqual(result.stream_id, "stream-123")
        self.assertEqual(len(result.versions), 2)
        self.assertEqual(result.versions[0].verification.anchor_status, "anchored")
        session.get.assert_called_once_with(
            "https://hydrogen.dclimate.net/api/datasets/eagle-temp/versions",
            params={
                "anchored": "true",
                "isCitable": "false",
                "versionLabel": "2026-05",
            },
            timeout=30,
        )

    def test_get_exact_version_returns_dataset_version(self):
        session = Mock()
        session.get.return_value = _mock_response(
            {
                "dataset": "eagle-temp",
                "cid": "cid-1",
                "commitId": "commit-1",
                "streamId": "stream-123",
                "publishedAt": "2026-05-02T14:26:43.131Z",
                "versionLabel": "2026-05",
                "isCitable": True,
                "retentionClass": "permanent",
                "verification": {"anchorStatus": "anchored"},
            }
        )

        result = get_exact_version("eagle-temp", "commit-1", session=session)

        self.assertIsInstance(result, DatasetVersion)
        self.assertEqual(result.dataset, "eagle-temp")
        self.assertEqual(result.commit_id, "commit-1")
        self.assertEqual(result.verification.anchor_status, "anchored")
        session.get.assert_called_once_with(
            "https://hydrogen.dclimate.net/api/datasets/eagle-temp/versions/commit-1",
            params=None,
            timeout=30,
        )

    def test_get_latest_metadata_uses_latest_dataset_endpoint(self):
        session = Mock()
        session.get.return_value = _mock_response(
            {
                "dataset": "eagle-temp",
                "cid": "cid-latest",
                "commitId": "commit-latest",
                "streamId": "stream-123",
                "verification": {"anchorStatus": "anchored"},
            }
        )

        result = get_latest_metadata("eagle-temp", session=session)

        self.assertEqual(result.cid, "cid-latest")
        session.get.assert_called_once_with(
            "https://hydrogen.dclimate.net/api/datasets/eagle-temp",
            params=None,
            timeout=30,
        )

    def test_get_citation_returns_citation_info(self):
        session = Mock()
        session.get.return_value = _mock_response(
            {
                "dataset": "eagle-temp",
                "streamId": "stream-123",
                "commitId": "commit-1",
                "cid": "cid-1",
                "publishedAt": "2026-05-02T14:26:43.131Z",
                "versionLabel": "2026-05",
                "isCitable": True,
                "retentionClass": "permanent",
                "citation": "citation text",
            }
        )

        result = get_citation("eagle-temp", commit_id="commit-1", session=session)

        self.assertIsInstance(result, CitationInfo)
        self.assertEqual(result.citation, "citation text")
        session.get.assert_called_once_with(
            "https://hydrogen.dclimate.net/api/datasets/eagle-temp/citation",
            params={"commitId": "commit-1"},
            timeout=30,
        )

    def test_filter_anchored_versions_keeps_only_anchored_entries(self):
        # Build versions through parsing so the verification model matches runtime code.
        parsed_versions = [
            DatasetVersion.from_api_payload(
                {
                    "dataset": "eagle-temp",
                    "cid": "cid-1",
                    "commitId": "commit-1",
                    "verification": {"anchorStatus": "anchored"},
                }
            ),
            DatasetVersion.from_api_payload(
                {
                    "dataset": "eagle-temp",
                    "cid": "cid-2",
                    "commitId": "commit-2",
                    "verification": {"anchorStatus": "pending"},
                }
            ),
        ]

        anchored = filter_anchored_versions(parsed_versions)

        self.assertEqual(len(anchored), 1)
        self.assertEqual(anchored[0].commit_id, "commit-1")

    def test_get_latest_anchored_version_picks_newest_timestamp(self):
        session = Mock()
        session.get.return_value = _mock_response(
            {
                "dataset": "eagle-temp",
                "streamId": "stream-123",
                "versions": [
                    {
                        "dataset": "eagle-temp",
                        "cid": "cid-1",
                        "commitId": "commit-1",
                        "timestamp": 100,
                        "verification": {"anchorStatus": "anchored"},
                    },
                    {
                        "dataset": "eagle-temp",
                        "cid": "cid-2",
                        "commitId": "commit-2",
                        "timestamp": 200,
                        "verification": {"anchorStatus": "anchored"},
                    },
                ],
            }
        )

        result = get_latest_anchored_version("eagle-temp", session=session)

        self.assertEqual(result.commit_id, "commit-2")
        session.get.assert_called_once_with(
            "https://hydrogen.dclimate.net/api/datasets/eagle-temp/versions",
            params={"anchored": "true"},
            timeout=30,
        )

    def test_get_latest_anchored_version_raises_when_no_anchored_versions_exist(self):
        session = Mock()
        session.get.return_value = _mock_response(
            {
                "dataset": "eagle-temp",
                "streamId": "stream-123",
                "versions": [],
            }
        )

        with self.assertRaisesRegex(
            ValueError, "No anchored versions found for dataset 'eagle-temp'"
        ):
            get_latest_anchored_version("eagle-temp", session=session)

    def test_build_gateway_url_joins_gateway_and_cid(self):
        self.assertEqual(
            build_gateway_url("bafytest", "https://gateway.example.com/"),
            "https://gateway.example.com/ipfs/bafytest",
        )

    def test_list_versions_from_stac_url_preserves_tritium_dataset_slug(self):
        session = Mock()
        session.get.return_value = _mock_response(
            {"dataset": "era5-temperature-2m-finalized", "versions": []}
        )

        result = list_versions_from_url(
            "https://tritium.dclimate.net/api/datasets/era5-temperature-2m-finalized/versions",
            anchored=True,
            session=session,
        )

        self.assertEqual(result.dataset, "era5-temperature-2m-finalized")
        session.get.assert_called_once_with(
            "https://tritium.dclimate.net/api/datasets/era5-temperature-2m-finalized/versions",
            params={"anchored": "true"},
            timeout=30,
        )

    def test_get_exact_version_from_stac_url_encodes_commit(self):
        session = Mock()
        session.get.return_value = _mock_response(
            {"dataset": "aigfs-wind-u", "cid": "cid-1"}
        )

        result = get_exact_version_from_url(
            "https://hydrogen.dclimate.net/api/datasets/aigfs-wind-u/versions",
            "commit/one",
            session=session,
        )

        self.assertEqual(result.cid, "cid-1")
        session.get.assert_called_once_with(
            "https://hydrogen.dclimate.net/api/datasets/aigfs-wind-u/versions/commit%2Fone",
            params=None,
            timeout=30,
        )

    def test_get_citation_from_stac_url_preserves_commit_query(self):
        session = Mock()
        citation_url = (
            "https://hydrogen.dclimate.net/api/datasets/aigfs-wind-u/citation"
            "?commitId=commit-1"
        )
        session.get.return_value = _mock_response(
            {
                "dataset": "aigfs-wind-u",
                "cid": "cid-1",
                "citation": "citation text",
            }
        )

        result = get_citation_from_url(citation_url, session=session)

        self.assertEqual(result.citation, "citation text")
        session.get.assert_called_once_with(citation_url, params=None, timeout=30)


if __name__ == "__main__":
    unittest.main()
