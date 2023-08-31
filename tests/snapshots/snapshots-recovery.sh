#!/bin/bash

set -ex

docker volume create snapshots
docker volume create tempdir
docker volume create storage

declare DOCKER_IMAGE_NAME=qdrant-snapshots
declare CONTAINER_NAME=qdrant-snapshots-container

docker buildx build --build-arg=PROFILE=ci --load ../../ --tag=$DOCKER_IMAGE_NAME


docker run \
    --rm -d \
    --network=host \
    -v snapshots:/qdrant/snapshots \
    -v tempdir:/qdrant/tempdir \
    -v storage:/qdrant/storage \
    -e QDRANT__STORAGE__TEMP_PATH=/qdrant/tempdir \
    --name ${CONTAINER_NAME} \
    $DOCKER_IMAGE_NAME


function clear() {
    docker rm -f ${CONTAINER_NAME}
    docker volume rm snapshots
    docker volume rm tempdir
    docker volume rm storage
}

trap clear EXIT

# Wait (up to ~30 seconds) for the service to start
declare retry=0
while [[ $(curl -sS localhost:6333 -w ''%{http_code}'' -o /dev/null) != 200 ]]; do
    if ((retry++ < 30)); then
        sleep 1
    else
        echo "Service failed to start in ~30 seconds" >&2
        exit 1
    fi
done

# Testing scenario:
# - Create collection, insert points and make snapshot
# - Download snapshot
# - Upload snapshot via URL
# - Upload snapshot as file

QDRANT_HOST='localhost:6333'

# Create collection
curl -X PUT "http://${QDRANT_HOST}/collections/test_collection" \
  -H 'Content-Type: application/json' \
  --fail -s \
  --data-raw '{
      "vectors": {
        "size": 4,
        "distance": "Dot"
      }
    }'

# Insert points
PAYLOAD=$( jq -n \
   '{ "points": [
      {"id": 1, "vector": [0.19, 0.81, 0.75, 0.11], "payload": {"city":  "London" }},
      {"id": 2, "vector": [0.05, 0.61, 0.76, 0.74], "payload": {"city":  "Berlin" }}
    ]}')

# insert points
curl -L -X PUT "http://$QDRANT_HOST/collections/test_collection/points?wait=true" \
  -H 'Content-Type: application/json' \
  --fail -s \
  --data-raw "$PAYLOAD" | jq

# Make snapshot

declare SNAPSHOT_NAME=$(curl -X POST "http://${QDRANT_HOST}/collections/test_collection/snapshots" -H 'Content-Type: application/json' --data-raw '{}' | jq -r '.result.name')

declare SNAPSHOT_URL="http://${QDRANT_HOST}/collections/test_collection/snapshots/${SNAPSHOT_NAME}"

# Download snapshot
curl -X GET ${SNAPSHOT_URL} -H 'Content-Type: application/json' --fail -s -o test_collection.snapshot

# Upload snapshot via URL
curl -X PUT "http://${QDRANT_HOST}/collections/test_collection_recovered_1/snapshots/recover" \
     -H 'Content-Type: application/json' \
     --fail -s -d "{\"location\": \"${SNAPSHOT_URL}\"}"

# Upload snapshot as file
curl -X POST "http://${QDRANT_HOST}/collections/test_collection_recovered_2/snapshots/upload" \
     -H 'Content-Type:multipart/form-data' \
     -F 'snapshot=@test_collection.snapshot'

# Check that all collections are present
curl -X GET "http://${QDRANT_HOST}/collections/test_collection_recovered_1" --fail

curl -X GET "http://${QDRANT_HOST}/collections/test_collection_recovered_2" --fail

