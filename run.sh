
DOCKER_VOLUMES="
--volume="./src/:/app/src/" \
"

docker run -i -t ${DOCKER_VOLUMES} romeo-rl:latest /bin/bash
