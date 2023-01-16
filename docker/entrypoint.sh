#!/bin/bash
# Create user
echo "in docker!"
groupadd -g 1008 metric-group
useradd -u 1008 -g 1008 metric-user -s /bin/bash
