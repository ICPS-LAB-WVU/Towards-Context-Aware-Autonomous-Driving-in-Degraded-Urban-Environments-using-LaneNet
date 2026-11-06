#!/bin/bash
source /environment.sh
dt-launchfile-init
dt-exec rosrun my_package vehicle_client.py
dt-launchfile-join
