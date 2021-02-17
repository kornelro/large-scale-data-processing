#!/usr/bin/env bash

docker build -t $2 $1
docker tag $2 kornelro/test_repo:$2
docker push kornelro/test_repo:$2