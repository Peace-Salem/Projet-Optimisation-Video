#!/usr/bin/env python3
"""
Solution for Hash Code 2017 Qualification Round - Streaming Videos
Using Gurobi for optimization
"""

import sys
import argparse
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
import gurobipy as gp
from gurobipy import GRB


@dataclass
class Video:
    id: int
    size: int


@dataclass
class Endpoint:
    id: int
    dc_latency: int
    cache_connections: Dict[int, int]  # cache_id -> latency


@dataclass
class Request:
    video_id: int
    endpoint_id: int
    count: int


@dataclass
class Cache:
    id: int
    capacity: int


class StreamingProblem:
    def __init__(self):
        self.videos: List[Video] = []
        self.endpoints: List[Endpoint] = []
        self.requests: List[Request] = []
        self.caches: List[Cache] = []
        self.total_requests = 0
        
    def read_input(self, filename: str):
        """Read input file in the specified format"""
        print(f"Reading input file: {filename}")
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Parse first line
        V, E, R, C, X = map(int, lines[0].split())
        print(f"Videos: {V}, Endpoints: {E}, Requests: {R}, Caches: {C}, Cache capacity: {X}")
        
        # Initialize caches
        self.caches = [Cache(i, X) for i in range(C)]
        
        # Parse video sizes
        video_sizes = list(map(int, lines[1].split()))
        self.videos = [Video(i, size) for i, size in enumerate(video_sizes)]
        
        line_idx = 2
        # Parse endpoints
        self.endpoints = []
        for endpoint_id in range(E):
            dc_latency, K = map(int, lines[line_idx].split())
            line_idx += 1
            
            endpoint = Endpoint(endpoint_id, dc_latency, {})
            
            for _ in range(K):
                cache_id, cache_latency = map(int, lines[line_idx].split())
                line_idx += 1
                endpoint.cache_connections[cache_id] = cache_latency
            
            self.endpoints.append(endpoint)
        
        # Parse requests
        self.requests = []
        self.total_requests = 0
        for _ in range(R):
            video_id, endpoint_id, count = map(int, lines[line_idx].split())
            line_idx += 1
            self.requests.append(Request(video_id, endpoint_id, count))
            self.total_requests += count
        
        print(f"Loaded {len(self.videos)} videos, {len(self.endpoints)} endpoints, "
              f"{len(self.requests)} request descriptions ({self.total_requests} total requests)")
    
    def create_model(self):
        """Create Gurobi optimization model"""
        print("\nCreating optimization model...")
        start_time = time.time()
        
        # Create model
        model = gp.Model("StreamingVideos")
        model.setParam('MIPGap', 0.005)  # 0.5% optimality gap
        
        # Decision variables
        # x[v][c] = 1 if video v is stored in cache c
        x = {}
        for v in range(len(self.videos)):
            for c in range(len(self.caches)):
                x[(v, c)] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c}")
        
        # y[r][c] = 1 if request r is served from cache c
        y = {}
        for r_idx, request in enumerate(self.requests):
            endpoint = self.endpoints[request.endpoint_id]
            for c in endpoint.cache_connections.keys():
                y[(r_idx, c)] = model.addVar(vtype=GRB.BINARY, name=f"y_{r_idx}_{c}")
        
        model.update()
        
        # Constraints
        print("Adding constraints...")
        
        # 1. Cache capacity constraints
        for c in range(len(self.caches)):
            cache_capacity = self.caches[c].capacity
            model.addConstr(
                gp.quicksum(self.videos[v].size * x[(v, c)] 
                          for v in range(len(self.videos))) <= cache_capacity,
                name=f"capacity_cache_{c}"
            )
        
        # 2. Each request can be served from at most one cache
        for r_idx, request in enumerate(self.requests):
            endpoint = self.endpoints[request.endpoint_id]
            connected_caches = list(endpoint.cache_connections.keys())
            
            # Sum over connected caches should be <= 1
            model.addConstr(
                gp.quicksum(y[(r_idx, c)] for c in connected_caches) <= 1,
                name=f"one_cache_per_request_{r_idx}"
            )
            
            # Link x and y: can only serve from cache if video is in cache
            for c in connected_caches:
                model.addConstr(
                    y[(r_idx, c)] <= x[(request.video_id, c)],
                    name=f"link_{r_idx}_{c}"
                )
        
        # Objective: maximize total time saved
        print("Setting objective function...")
        objective_terms = []
        
        for r_idx, request in enumerate(self.requests):
            endpoint = self.endpoints[request.endpoint_id]
            dc_latency = endpoint.dc_latency
            
            # If served from data center: time saved = 0
            # If served from cache c: time saved = (dc_latency - cache_latency) * request.count
            
            for c, cache_latency in endpoint.cache_connections.items():
                time_saved_per_request = dc_latency - cache_latency
                total_time_saved = time_saved_per_request * request.count
                objective_terms.append(total_time_saved * y[(r_idx, c)])
        
        model.setObjective(gp.quicksum(objective_terms), GRB.MAXIMIZE)
        
        model.update()
        
        # Export MPS file
        print("Exporting MPS file...")
        model.write("videos.mps")
        
        elapsed = time.time() - start_time
        print(f"Model created in {elapsed:.2f} seconds")
        
        return model, x, y
    
    def solve(self, model):
        """Solve the optimization model"""
        print("\nSolving optimization model...")
        start_time = time.time()
        
        model.optimize()
        
        elapsed = time.time() - start_time
        print(f"Solution found in {elapsed:.2f} seconds")
        
        if model.status == GRB.OPTIMAL:
            print(f"Optimal solution found with objective: {model.objVal}")
        elif model.status == GRB.TIME_LIMIT:
            print(f"Solution found within time limit with gap: {model.MIPGap:.4%}")
        else:
            print(f"Solution status: {model.status}")
    
    def write_output(self, x_vars, filename="videos.out"):
        """Write solution in required format"""
        print(f"\nWriting solution to {filename}")
        
        # Extract which videos are in which caches
        cache_contents = {}
        for (v, c), var in x_vars.items():
            if var.X > 0.5:  # Video is in cache
                if c not in cache_contents:
                    cache_contents[c] = []
                cache_contents[c].append(v)
        
        # Write output file
        with open(filename, 'w') as f:
            # Number of cache servers with videos
            f.write(f"{len(cache_contents)}\n")
            
            # Write cache contents
            for cache_id in sorted(cache_contents.keys()):
                videos = cache_contents[cache_id]
                line = f"{cache_id} " + " ".join(map(str, videos))
                f.write(line + "\n")
        
        print(f"Written {len(cache_contents)} cache servers with videos")


def main():
    parser = argparse.ArgumentParser(description="Solve Hash Code 2017 Streaming Videos problem")
    parser.add_argument("dataset", help="Path to input dataset file")
    args = parser.parse_args()
    
    # Create problem instance
    problem = StreamingProblem()
    
    # Read input
    problem.read_input(args.dataset)
    
    # Create and solve model
    model, x_vars, y_vars = problem.create_model()
    problem.solve(model)
    
    # Write output
    problem.write_output(x_vars)
    
    # Calculate and display score
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        # Convert objective (maximized time saved in ms) to score
        total_time_saved_ms = model.objVal
        score = (total_time_saved_ms * 1000) / problem.total_requests
        print(f"\nEstimated score: {int(score)}")


if __name__ == "__main__":
    main()