#!/bin/bash

for geo in USA AUS BEL SWE ZMB; do
    python sim.py \
     --geo $geo \
     --start-index 100 \
     --steps 365
done
