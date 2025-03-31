from functools import lru_cache

# There are 7 cities labeled 1 through 7.
# The symmetric distance matrix (with 14's replaced by 12's):
# Row i and column j represent the distance from City (i+1) to City (j+1)
dist = [
    [0, 12, 10,  0,  0,  0, 12],
    [12, 0,  8, 12,  0,  0,  0],
    [10, 8,  0, 11,  3,  0,  9],
    [0, 12, 11,  0, 11, 10,  0],
    [0,  0,  3, 11,  0,  6,  7],
    [0,  0,  0, 10,  6,  0,  9],
    [12, 0,  9,  0,  7,  9,  0]
]

n = len(dist)             # Total number of cities (7)
FULL_MASK = (1 << n) - 1  # Bitmask representing that all cities have been visited
START_CITY = 0            # City 1 (0-indexed) is the start and end point

# Dictionary to store the next city choices for reconstructing the optimal path.
parent = {}

@lru_cache(maxsize=None)
def tsp(mask, pos):
    """
    Compute the minimum TSP tour cost using dynamic programming with bitmasking.
    
    Args:
        mask (int): A bitmask representing visited cities.
        pos (int): The current city index.
        
    Returns:
        int: The minimum cost to visit all remaining cities from the current state and return to the start.
    """
    # Base case: all cities have been visited; return cost to return to the start.
    if mask == FULL_MASK:
        return dist[pos][START_CITY] if dist[pos][START_CITY] > 0 else float('inf')
    
    min_cost = float('inf')
    best_next = None
    
    # Try all unvisited cities.
    for next_city in range(n):
        if mask & (1 << next_city) == 0 and dist[pos][next_city] > 0:
            new_mask = mask | (1 << next_city)
            cost = dist[pos][next_city] + tsp(new_mask, next_city)
            if cost < min_cost:
                min_cost = cost
                best_next = next_city
    
    # Store the best next city for path reconstruction.
    parent[(mask, pos)] = best_next
    return min_cost

def reconstruct_path():
    """
    Reconstruct the optimal TSP route using stored decisions.
    
    Returns:
        list: The list of cities (1-indexed) representing the optimal route.
    """
    mask = (1 << START_CITY)  # Starting with only the start city visited.
    pos = START_CITY
    path = [pos]              # Start path (0-indexed)
    
    # Follow the stored best choices until all cities are visited.
    while True:
        next_city = parent.get((mask, pos))
        if next_city is None:
            break
        path.append(next_city)
        mask |= (1 << next_city)
        pos = next_city
        if mask == FULL_MASK:
            break
    
    # Finally, add the starting city to complete the tour.
    path.append(START_CITY)
    # Convert route to 1-indexed for clarity.
    return [city + 1 for city in path]

# Compute the optimal cost and path starting from City 1 (index 0)
initial_mask = (1 << START_CITY)
optimal_cost = tsp(initial_mask, START_CITY)
optimal_route = reconstruct_path()

print(f"Minimum cost of TSP tour starting and ending at City 1: {optimal_cost}")
print("Optimal route:", " -> ".join(map(str, optimal_route)))