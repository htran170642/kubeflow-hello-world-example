CONTAINER_NAME=pytorch-mnist-tensorflow
PROJECT_ID=170642
TAG_NAME=latest

docker build -t ${CONTAINER_NAME} .
docker tag ${CONTAINER_NAME} ${PROJECT_ID}/${CONTAINER_NAME}:${TAG_NAME}
docker push ${PROJECT_ID}/${CONTAINER_NAME}:${TAG_NAME}
