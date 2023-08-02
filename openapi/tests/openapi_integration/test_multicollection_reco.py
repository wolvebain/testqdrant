import pytest

from .helpers.collection_setup import basic_collection_setup, drop_collection
from .helpers.helpers import request_with_validation

collection_name = 'test_collection_reco'
collection_name2 = 'test_collection_reco2'


@pytest.fixture(autouse=True, scope="module")
def setup():
    basic_collection_setup(collection_name=collection_name)
    yield
    drop_collection(collection_name=collection_name)


def test_recommend_with_wrong_vector_size():
    response = request_with_validation(
        api='/collections/{collection_name}',
        method="DELETE",
        path_params={'collection_name': collection_name2},
    )
    assert response.ok

    response = request_with_validation(
        api='/collections/{collection_name}',
        method="PUT",
        path_params={'collection_name': collection_name2},
        body={
            "vectors": {
                "other": {
                    "size": 5,
                    "distance": "Dot"
                }
            }
        }
    )
    assert response.ok

    response = request_with_validation(
        api='/collections/{collection_name}/points',
        method="PUT",
        path_params={'collection_name': collection_name2},
        query_params={'wait': 'true'},
        body={
            "points": [
                {
                    "id": 1,
                    "vector": {"other": [1.0, 0.0, 0.0, 0.0, 0.0]},
                },
                {
                    "id": "00000000-0000-0000-0000-000000000000",
                    "vector": {"other": [0.0, 1.0, 0.0, 0.0, 0.0]},
                }
            ]
        }
    )
    assert response.ok, response.text

    response = request_with_validation(
        api='/collections/{collection_name}/points/recommend',
        method="POST",
        path_params={'collection_name': collection_name},
        body={
            "limit": 3,
            "positive": [1],
            "with_vector": False,
            "with_payload": True,
            "lookup_from": {
                "collection": collection_name2,
                "vector": "other"
            }
        }
    )
    assert response.status_code == 400, response.text


def test_recommend_from_another_collection():
    # Create another collection with the same vector size.
    # Use vectors from the second collection to search in the first collection.

    response = request_with_validation(
        api='/collections/{collection_name}',
        method="DELETE",
        path_params={'collection_name': collection_name2},
    )
    assert response.ok

    response = request_with_validation(
        api='/collections/{collection_name}',
        method="PUT",
        path_params={'collection_name': collection_name2},
        body={
            "vectors": {
                "other": {
                    "size": 4,
                    "distance": "Dot"
                }
            }
        }
    )
    assert response.ok

    response = request_with_validation(
        api='/collections/{collection_name}/points',
        method="PUT",
        path_params={'collection_name': collection_name2},
        query_params={'wait': 'true'},
        body={
            "points": [
                {
                    "id": 1,
                    "vector": {"other": [1.0, 0.0, 0.0, 0.0]},
                },
                {
                    "id": "00000000-0000-0000-0000-000000000000",
                    "vector": {"other": [0.0, 1.0, 0.0, 0.0]},
                }
            ]
        }
    )
    assert response.ok, response.text

    response = request_with_validation(
        api='/collections/{collection_name}/points/recommend',
        method="POST",
        path_params={'collection_name': collection_name},
        body={
            "limit": 3,
            "positive": [1],
            "with_vector": False,
            "with_payload": True,
            "lookup_from": {
                "collection": collection_name2,
                "vector": "other"
            }
        }
    )
    assert response.ok, response.text
    assert len(response.json()['result']) == 3
    # vector with the largest 1st element
    assert response.json()['result'][0]['id'] == 8

    response = request_with_validation(
        api='/collections/{collection_name}/points/recommend/batch',
        method="POST",
        path_params={'collection_name': collection_name},
        body={
            "searches": [
                {
                    "limit": 3,
                    "positive": [1],
                    "with_vector": False,
                    "with_payload": True,
                    "lookup_from": {
                        "collection": collection_name2,
                        "vector": "other"
                    }
                },
                {
                    "limit": 3,
                    "positive": ["00000000-0000-0000-0000-000000000000"],
                    "with_vector": False,
                    "with_payload": True,
                    "lookup_from": {
                        "collection": collection_name2,
                        "vector": "other"
                    }
                },
                {
                    "limit": 3,
                    "positive": [1],
                    "with_vector": False,
                    "with_payload": True,
                }
            ]
        }
    )
    assert response.ok, response.text
    assert len(response.json()['result']) == 3
    assert len(response.json()['result'][0]) == 3
    assert len(response.json()['result'][1]) == 3
    assert len(response.json()['result'][2]) == 3
    # vector with the largest 1st element
    assert response.json()['result'][0][0]['id'] == 8
    # vector with the largest 2nd element
    assert response.json()['result'][1][0]['id'] == 7

    response = request_with_validation(
        api='/collections/{collection_name}/points/recommend',
        method="POST",
        path_params={'collection_name': collection_name},
        body={
            "limit": 3,
            "positive": [1],
            "lookup_from": {
                "collection": "unknown_collection",
                "vector": "other"
            }
        }
    )
    assert response.status_code == 404

    response = request_with_validation(
        api='/collections/{collection_name}/points/recommend',
        method="POST",
        path_params={'collection_name': collection_name},
        body={
            "limit": 3,
            "positive": [1],
            "lookup_from": {
                "collection": collection_name2,
                "vector": "unknown_vector"
            }
        }
    )
    assert response.status_code == 400, response.text

    response = request_with_validation(
        api='/collections/{collection_name}/points/recommend',
        method="POST",
        path_params={'collection_name': collection_name},
        body={
            "limit": 3,
            "positive": [2],
            "lookup_from": {
                "collection": collection_name2,
                "vector": "unknown_vector"
            }
        }
    )
    assert response.status_code == 404, response.text

    response = request_with_validation(
        api='/collections/{collection_name}',
        method="DELETE",
        path_params={'collection_name': collection_name2},
    )
    assert response.ok
