#!/bin/bash

set -e

trap 'trap - SIGTERM && kill -- -$$ && kill $(jobs -p)' SIGINT SIGTERM EXIT

# FIXME real example data

python live_server.py merlin.toml acquisition_parameters_SSPED.toml &
python live_virtual_detectors.py "SSPED live server demo" &
libertem-live-mib-sim ~/er-c-data/adhoc/livespedescan/demo/SSPED/230523_Fe-Joakim_D_Scan9_excerpt/default.hdr
