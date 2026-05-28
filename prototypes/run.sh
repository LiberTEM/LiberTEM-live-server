#!/bin/bash

set -e

trap 'trap - SIGTERM && kill -- -$$ && kill $(jobs -p)' SIGINT SIGTERM EXIT

# FIXME real example data

python live_server.py merlin.toml acquisition_parameters_SSPED.toml &
python live_virtual_detectors.py &
libertem-live-mib-sim --dwelltime 300 --cached=MEM ~/er-c-data-Nextcloud/ER-C-Data/adhoc/libertem/libertem-test-data/20200518\ 165148/default.hdr
