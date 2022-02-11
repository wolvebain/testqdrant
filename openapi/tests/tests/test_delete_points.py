import pytest

from ..openapi_integration.helpers import request_with_validation
from ..openapi_integration.collection_setup import basic_collection_setup, drop_collection

collection_name = 'test_collection_delete'


@pytest.fixture(autouse=True)
def setup():
    basic_collection_setup(collection_name=collection_name)
    yield
    drop_collection(collection_name=collection_name)


def test_points_retrieve():
    # delete point by filter (has_id)
    response = request_with_validation(
        api='/collections/{name}/points/delete',
        method="POST",
        path_params={'name': collection_name},
        query_params={'wait': 'true'},
        body={
            "filter": {
                "must": [
                    {"has_id": [5]}
                ]
            }
        }
    )
    assert response.ok

    # quantity check if the above point id was deleted
    response = request_with_validation(
        api='/collections/{name}',
        method="GET",
        path_params={'name': collection_name},
    )
    assert response.ok
    assert response.json()['result']['vectors_count'] == 5

    response = request_with_validation(
        api='/collections/{name}/points/delete',
        method="POST",
        path_params={'name': collection_name},
        query_params={'wait': 'true'},
        body={
            "points": [1, 2, 3, 4]
        }
    )
    assert response.ok

    # quantity check if the above point id was deleted
    response = request_with_validation(
        api='/collections/{name}',
        method="GET",
        path_params={'name': collection_name},
    )
    assert response.ok
    assert response.json()['result']['vectors_count'] == 1
