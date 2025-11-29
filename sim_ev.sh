#!/bin/bash

for geo in USA AUS BEL SWE ZMB; do
    python sim_ev.py \
     --geo $geo \
     --start-index 100 \
     --steps 365 \
     --generations 30
done
