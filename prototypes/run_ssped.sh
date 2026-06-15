#!/bin/bash

set -e

trap 'trap - SIGTERM && kill -- -$$ && kill $(jobs -p)' SIGINT SIGTERM EXIT

# FIXME real example data

python live_server.py merlin.toml acquisition_parameters_SSPED.toml &
python live_virtual_detectors.py "SSPED live server demo" &
libertem-live-mib-sim ~/er-c-data/adhoc/livespedescan/SSPED/SPED_256x256_22x22_10186nm_NBD_a5_spot1nm_CL20cm_125msExp_3000msFB_subframing_x8_pivotoff_01/default.hdr
