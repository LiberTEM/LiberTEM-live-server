#!/bin/bash

set -e

trap 'trap - SIGTERM && kill -- -$$ && kill $(jobs -p)' SIGINT SIGTERM EXIT

# FIXME real example data

python live_server.py merlin.toml acquisition_parameters_SSPED.toml &
python live_virtual_detectors.py "SSPED live server demo" &
libertem-live-mib-sim ~/er-c-data/adhoc/livespedescan/SSPED/by_sivert/20230523_200538/230523_Fe-Joakim_D_Scan9.hdr
