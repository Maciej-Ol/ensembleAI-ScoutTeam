#!/bin/bash

for i in {1..7}; do
    python3 defense_submit.py "$i"  # Pass the parameter to the Python script
    echo "I have finished defense_submit successfully!"
    sleep 4000  # Sleep for 3600 seconds (1 hour)
done

