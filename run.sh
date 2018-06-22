IMAGE_NAME=$1
DIR_PATH=$(cd $(dirname $0); pwd)

docker run --rm -it \
  -v $(pwd):${DIR_PATH} \
  -w ${DIR_PATH} \
  "${IMAGE_NAME}" bash
