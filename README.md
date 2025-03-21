# Traveling Salesman Problem (TSP) Solver

This Python implementation solves the Traveling Salesman Problem using dynamic programming with a bitmask approach. The code finds the optimal route through 7 cities that minimizes the total travel distance, starting and ending at City 1.

## Problem Description

The program works with a specific distance matrix for 7 cities (labeled 1 through 7), where:
- Each entry `dist[i][j]` represents the distance from City (i+1) to City (j+1)
- The matrix is symmetric (distance from A to B equals distance from B to A)
- A value of 0 indicates no direct connection between cities
- Some entries with original value 14 have been replaced with 12

## Implementation Details

The solution uses:
- **Dynamic Programming** with memoization (via `@lru_cache`)
- **Bitmask** representation to track visited cities efficiently
- A **recursive approach** that explores all possible paths

### Key Components

1. **Distance Matrix (`dist`)**: A 7×7 matrix storing distances between cities
2. **TSP Function**: Recursively computes the minimum cost tour
3. **Path Reconstruction**: Reconstructs the optimal path using stored decisions
4. **Bitmask Representation**: Uses bits to track which cities have been visited

## Complexity Analysis

- **Time Complexity**: O(n²·2ⁿ) where n is the number of cities
- **Space Complexity**: O(n·2ⁿ) for storing the memoization table

## Usage

Simply run the Python script:

```bash
python tsp_solver.py
```

The output shows:
1. The minimum cost of the TSP tour
2. The optimal route (sequence of cities)

## Example Output

```
Minimum cost of TSP tour starting and ending at City 1: [optimal cost value]
Optimal route: 1 -> [sequence of cities] -> 1
```

## Implementation Notes

- The implementation uses 0-indexed cities internally but displays 1-indexed cities in the output
- The `parent` dictionary tracks the next city in the optimal path for path reconstruction
- The special case where a city cannot be reached directly is handled by returning infinity
- The `lru_cache` decorator significantly improves performance by avoiding redundant calculations
