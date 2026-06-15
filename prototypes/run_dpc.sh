#!/bin/bash

set -e

trap 'trap - SIGTERM && kill -- -$$ && kill $(jobs -p)' SIGINT SIGTERM EXIT

# FIXME real example data

python live_server.py merlin.toml acquisition_parameters_STEM-DPC.toml &
python live_virtual_detectors.py "DPC live server demo" &
libertem-live-mib-sim ~/er-c-data/adhoc/livespedescan/DPC/001_LMSTEM_x15k_512x512_5_10msExp_100msFB_nopres/default.hdr
