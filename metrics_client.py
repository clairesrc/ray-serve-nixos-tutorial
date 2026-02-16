import requests
import sys
import time

# Default Ray metrics port is 8080
METRICS_URL = "http://localhost:8080/metrics"

def get_metrics():
    try:
        response = requests.get(METRICS_URL)
        response.raise_for_status()
        return response.text
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to {METRICS_URL}. Is Ray running?")
        return None

def parse_and_print_serve_metrics(metrics_data):
    print(f"\n--- Ray Serve Metrics Snapshot ({time.strftime('%H:%M:%S')}) ---")
    
    # We are looking for specific Serve metrics
    interesting_metrics = [
        "serve_deployment_request_latency_ms",
        "serve_deployment_queuing_latency_ms",
        "serve_num_router_requests",
        "serve_num_outstanding_requests",
        "serve_deployment_replica_healthy",
        "ray_serve_deployment_request_latency_ms_count",
        "ray_serve_deployment_request_latency_ms_sum",
    ]
    
    found_any = False
    for line in metrics_data.splitlines():
        if line.startswith("#"):
            continue
        
        # Check if line contains any of our interesting metrics
        for metric in interesting_metrics:
            if metric in line:
                print(line)
                found_any = True
                break
    
    if not found_any:
        print("No Serve-specific metrics found yet. (Deployments might be idle or starting up)")

if __name__ == "__main__":
    print(f"Fetching metrics from {METRICS_URL}...")
    data = get_metrics()
    if data:
        parse_and_print_serve_metrics(data)
        print("\nTip: In production, Prometheus scrapes this endpoint to populate Grafana dashboards.")
