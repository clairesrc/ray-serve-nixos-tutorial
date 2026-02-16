#!/bin/bash
set -e

echo "--- Stopping any existing Ray instances ---"
ray stop --force || true

echo "--- Starting Head Node ---"
# Head node runs the Global Control Service (GCS) and the Dashboard.
# We bind it to port 6379 (default).
ray start --head --port=6379 --include-dashboard=true --dashboard-host=0.0.0.0 --num-cpus=2 --disable-usage-stats

echo "--- Starting Worker Node ---"
# Worker node connects to the Head Node via the Redis address (GCS).
# We specify a different object manager port to avoid conflicts on localhost.
# On different machines, you wouldn't need to specify ports manually usually.
ray start --address=localhost:6379 --num-cpus=2 --object-manager-port=8076 --disable-usage-stats

echo "--- Verifying Cluster Config ---"
# We run a small python script that connects to the cluster and prints resources.
python -c "
import ray
import time

# Connect to the existing cluster
ray.init(address='auto')

print('Waiting for nodes to register...')
time.sleep(5)

resources = ray.cluster_resources()
print(f'Cluster Resources: {resources}')

# Check if we have 4 CPUs total (2 from head + 2 from worker)
total_cpus = resources.get('CPU', 0)
print(f'Total CPUs: {total_cpus}')

if total_cpus >= 4:
    print('SUCCESS: Cluster is up with Head and Worker nodes!')
else:
    print('FAILURE: Expected at least 4 CPUs.')
    exit(1)
"

echo "--- Cleaning up ---"
ray stop --force
echo "Cluster simulation complete."
