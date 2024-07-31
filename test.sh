#!/bin/bash

rm -rf logs/connect_x.log && sudo docker run -it  -v ./connect_x:/app/connect_x -v ./logs:/app/logs -v ./runs:/app/runs -v ./models:/app/models connect-x test $@