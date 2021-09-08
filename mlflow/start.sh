#!/bin/sh

echo 'starting mlflow model'
mlflow --version
mlflow models serve -m /app/model --no-conda --port 8080 --host 0.0.0.0 --workers 1