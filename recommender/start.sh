#!/bin/sh

echo 'starting recommender'

uwsgi --socket 0.0.0.0:8081 --protocol=http -w wsgi:app
