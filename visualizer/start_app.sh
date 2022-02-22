#!/bin/bash

cd visualizer/server
source setup.sh
flask run --host 0.0.0.0 --port 52000 &
echo "Started the server in the background listening at localhost:52000"

cd ../client
echo "Building the client..."
# npm install --save-dev webpack --ignore-scripts
# npm run build
cd dist
python -m http.server 8001
echo "Started the client at localhost:8001/sysa.html"