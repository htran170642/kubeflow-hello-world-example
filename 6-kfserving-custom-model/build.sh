CONTAINER_NAME=kfserving-custom-model
PROJECT_ID=170642
TAG_NAME=latest

docker build -t ${CONTAINER_NAME} ./model-server
docker tag ${CONTAINER_NAME} ${PROJECT_ID}/${CONTAINER_NAME}:${TAG_NAME}
docker push ${PROJECT_ID}/${CONTAINER_NAME}:${TAG_NAME}
