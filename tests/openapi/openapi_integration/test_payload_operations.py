import pytest

from .helpers.collection_setup import basic_collection_setup, drop_collection
from .helpers.helpers import request_with_validation

collection_name = 'test_collection_payload'


@pytest.fixture(autouse=True)
def setup(on_disk_vectors):
    basic_collection_setup(collection_name=collection_name, on_disk_vectors=on_disk_vectors)
    yield
    drop_collection(collection_name=collection_name)


def test_payload_operations():
    # create payload
    response = request_with_validation(
        api='/collections/{collection_name}/points/payload',
        method="POST",
        path_params={'collection_name': collection_name},
        query_params={'wait': 'true'},
        body={
            "payload": {"test_payload": "keyword"},
            "points": [6]
        }
    )
    assert response.ok

    # check payload
    response = request_with_validation(
        api='/collections/{collection_name}/points/{id}',
        method="GET",
        path_params={'collection_name': collection_name, 'id': 6},
    )
    assert response.ok
    assert len(response.json()['result']['payload']) == 1

    # clean payload by filter
    response = request_with_validation(
        api='/collections/{collection_name}/points/payload/clear',
        method="POST",
        path_params={'collection_name': collection_name},
        query_params={'wait': 'true'},
        body={
            "filter": {
                "must": [
                    {"has_id": [6]}
                ]
            }
        }
    )
    assert response.ok

    # check payload
    response = request_with_validation(
        api='/collections/{collection_name}/points/{id}',
        method="GET",
        path_params={'collection_name': collection_name, 'id': 6},
    )
    assert response.ok
    assert len(response.json()['result']['payload']) == 0

    # create payload
    response = request_with_validation(
        api='/collections/{collection_name}/points/payload',
        method="POST",
        path_params={'collection_name': collection_name},
        query_params={'wait': 'true'},
        body={
            "payload": {"test_payload": "keyword"},
            "points": [6]
        }
    )
    assert response.ok

    # delete payload by id
    response = request_with_validation(
        api='/collections/{collection_name}/points/payload/delete',
        method="POST",
        path_params={'collection_name': collection_name},
        query_params={'wait': 'true'},
        body={
            "keys": ["test_payload"],
            "points": [6]
        }
    )
    assert response.ok

    # check payload
    response = request_with_validation(
        api='/collections/{collection_name}/points/{id}',
        method="GET",
        path_params={'collection_name': collection_name, 'id': 6},
    )
    assert response.ok
    assert len(response.json()['result']['payload']) == 0

    #
    # test PUT vs POST of the payload - set vs overwrite
    #
    response = request_with_validation(
        api='/collections/{collection_name}/points/payload',
        method="PUT",
        path_params={'collection_name': collection_name},
        body={
            "payload": {"key1": "aaa", "key2": "bbb"},
            "points": [6]
        }
    )
    assert response.ok

    # check payload
    response = request_with_validation(
        api='/collections/{collection_name}/points/{id}',
        method="GET",
        path_params={'collection_name': collection_name, 'id': 6},
    )
    assert response.ok
    assert len(response.json()['result']['payload']) == 2

    response = request_with_validation(
        api='/collections/{collection_name}/points/payload',
        method="POST",
        path_params={'collection_name': collection_name},
        body={
            "payload": {"key1": "ccc"},
            "points": [6]
        }
    )
    assert response.ok

    # check payload
    response = request_with_validation(
        api='/collections/{collection_name}/points/{id}',
        method="GET",
        path_params={'collection_name': collection_name, 'id': 6},
    )
    assert response.ok
    assert len(response.json()['result']['payload']) == 2
    assert response.json()['result']['payload']["key1"] == "ccc"
    assert response.json()['result']['payload']["key2"] == "bbb"

    response = request_with_validation(
        api='/collections/{collection_name}/points/payload',
        method="PUT",
        path_params={'collection_name': collection_name},
        body={
            "payload": {"key2": "eee"},
            "points": [6]
        }
    )
    assert response.ok

    # check payload
    response = request_with_validation(
        api='/collections/{collection_name}/points/{id}',
        method="GET",
        path_params={'collection_name': collection_name, 'id': 6},
    )
    assert response.ok
    assert len(response.json()['result']['payload']) == 1
    assert response.json()['result']['payload']["key2"] == "eee"

    #
    # Check set and update of payload by filter
    #
    response = request_with_validation(
        api='/collections/{collection_name}/points/payload',
        method="POST",
        path_params={'collection_name': collection_name},
        body={
            "payload": {"key4": "aaa"},
            "points": [1, 2, 3]
        }
    )
    assert response.ok

    response = request_with_validation(
        api='/collections/{collection_name}/points/scroll',
        method="POST",
        path_params={'collection_name': collection_name},
        body={
            "filter": {
                "must": [
                    {
                        "key": "key4",
                        "match": {
                            "value": "aaa"
                        }
                    }
                ]
            }
        }
    )
    assert response.ok
    assert len(response.json()['result']['points']) == 3

    response = request_with_validation(
        api='/collections/{collection_name}/points/payload',
        method="POST",
        path_params={'collection_name': collection_name},
        body={
            "payload": {"key5": "bbb"},
            "filter": {
                "must": [
                    {
                        "key": "key4",
                        "match": {
                            "value": "aaa"
                        }
                    }
                ]
            }
        }
    )
    assert response.ok

    response = request_with_validation(
        api='/collections/{collection_name}/points/scroll',
        method="POST",
        path_params={'collection_name': collection_name},
        body={
            "filter": {
                "must": [
                    {
                        "key": "key5",
                        "match": {
                            "value": "bbb"
                        }
                    }
                ]
            }
        }
    )
    assert response.ok
    assert len(response.json()['result']['points']) == 3

    # set property of payload by empty key
    # response = request_with_validation(
        # api='/collections/{collection_name}/points/payload',
        # method="POST",
        # path_params={'collection_name': collection_name},
        # body={
            # "payload": {"key6": "xxx"},
            # "points": [1],
            # "key": ""
        # }
    # )
    # assert not response.ok

    # response = request_with_validation(
        # api='/collections/{collection_name}/points/{id}',
        # method="GET",
        # path_params={'collection_name': collection_name, 'id': 1},
    # )
    # assert response.ok
    # assert len(response.json()['result']['payload']) == 3

    # set property of payload with top level
    response = request_with_validation(
        api='/collections/{collection_name}/points/payload',
        method="POST",
        path_params={'collection_name': collection_name},
        body={
            "payload": {"key6": {"subkey": "xxx", "subkey2": {"subkey3": "xxx"}}},
            "points": [1],
        }
    )
    assert response.ok

    # set property of payload with top level key
    response = request_with_validation(
        api='/collections/{collection_name}/points/payload',
        method="POST",
        path_params={'collection_name': collection_name},
        body={
            "payload": {"subkey": "yyy"},
            "points": [1],
            "key": "key6",
        }
    )
    assert response.ok

    response = request_with_validation(
        api='/collections/{collection_name}/points/{id}',
        method="GET",
        path_params={'collection_name': collection_name, 'id': 1},
    )
    assert response.ok
    assert len(response.json()['result']['payload']) == 4
    assert response.json()['result']['payload']["key6"]["subkey"] == "yyy"

    # set property of payload with nested key
    response = request_with_validation(
        api='/collections/{collection_name}/points/payload',
        method="POST",
        path_params={'collection_name': collection_name},
        body={
            "payload": {"subkey3": "yyy"},
            "points": [1],
            "key": "key6.subkey2",
        }
    )
    assert response.ok

    response = request_with_validation(
        api='/collections/{collection_name}/points/{id}',
        method="GET",
        path_params={'collection_name': collection_name, 'id': 1},
    )
    assert response.ok
    assert len(response.json()['result']['payload']) == 4
    assert response.json()['result']['payload']["key6"]["subkey2"]["subkey3"] == "yyy"

    response = request_with_validation(
        api='/collections/{collection_name}/points/payload/delete',
        method="POST",
        path_params={'collection_name': collection_name},
        body={
            "keys": ["key5"],
            "filter": {
                "must": [
                    {
                        "key": "key5",
                        "match": {
                            "value": "bbb"
                        }
                    }
                ]
            }
        }
    )
    assert response.ok

    response = request_with_validation(
        api='/collections/{collection_name}/points/scroll',
        method="POST",
        path_params={'collection_name': collection_name},
        body={
            "filter": {
                "must": [
                    {
                        "key": "key5",
                        "match": {
                            "value": "bbb"
                        }
                    }
                ]
            }
        }
    )
    assert response.ok
    assert len(response.json()['result']['points']) == 0
