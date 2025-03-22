import numpy as np
import matplotlib.pyplot as plt

# Adjusted city positions based on the image structure
city_positions = np.array([
    [0, 3],    # City 1 (Start)
    [2, 4],    # City 2
    [1.5, 2.8], # City 3
    [3, 3.5],  # City 4
    [2.5, 2],  # City 5
    [3.5, 1],  # City 6
    [0.5, 1.5] # City 7
])

# Adjacency matrix from the provided image
city_distances = np.array([
    [0, 12, 10, 0, 0, 0, 12],
    [12, 0, 8, 12, 0, 0, 0],
    [10, 8, 0, 11, 3, 0, 9],
    [0, 12, 11, 0, 11, 10, 0],
    [0, 0, 3, 11, 0, 6, 7],
    [0, 0, 0, 10, 6, 0, 9],
    [12, 0, 9, 0, 7, 9, 0]
])

num_cities = city_positions.shape[0]
num_neurons = num_cities * 5  # More neurons for better accuracy

# Initialize neurons randomly around city positions
neurons = np.random.rand(num_neurons, 2) * 5  

# Training parameters (fine-tuned for better convergence)
num_iterations = 5000  # More iterations for better learning
learning_rate = 0.6    # Adjusted learning rate
initial_radius = num_neurons // 3

# Training the SOM
for i in range(num_iterations):
    city = city_positions[np.random.randint(0, num_cities)]

    distances = np.linalg.norm(neurons - city, axis=1)
    winner_index = np.argmin(distances)

    radius = initial_radius * np.exp(-i / (num_iterations / np.log(initial_radius + 1)))
    influence = np.exp(-distances**2 / (2 * (max(radius, 1e-6) ** 2)))

    neurons += learning_rate * influence[:, np.newaxis] * (city - neurons)

# Match each city to its nearest neuron
ordered_indices = []
for city in city_positions:
    nearest_neuron = np.argmin(np.linalg.norm(neurons - city, axis=1))
    ordered_indices.append((nearest_neuron, city))

# Ensure order and cyclic path
ordered_indices = sorted(ordered_indices, key=lambda x: x[0])
ordered_cities = np.array([city for _, city in ordered_indices])
ordered_city_indices = [np.where((city_positions == city).all(axis=1))[0][0] for city in ordered_cities]

# Close the loop (return to start city)
ordered_cities = np.vstack([ordered_cities, ordered_cities[0]])
ordered_city_indices.append(ordered_city_indices[0])

# Calculate total distance using the adjacency matrix
total_distance = 0
for i in range(len(ordered_city_indices) - 1):
    city_a = ordered_city_indices[i]
    city_b = ordered_city_indices[i + 1]
    total_distance += city_distances[city_a][city_b]

# Plot results
plt.figure(figsize=(6, 6))
plt.plot(ordered_cities[:, 0], ordered_cities[:, 1], 'b-o', label="Route Path")
plt.scatter(city_positions[:, 0], city_positions[:, 1], c='r', marker='o', label="Cities")

# Label cities
for i, (x, y) in enumerate(city_positions):
    plt.text(x, y, f"{i+1}", fontsize=12, verticalalignment='bottom', horizontalalignment='right')

# Display total distance on the plot (placed at the bottom)
plt.text(min(city_positions[:, 0]), min(city_positions[:, 1]) - 0.5, 
         f"Total Distance: {total_distance:.2f}", 
         fontsize=12, color='blue', bbox=dict(facecolor='white', alpha=0.7))

plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.title("Optimized TSP Route Using SOM")
plt.legend()
plt.grid()
plt.show()

print("Total Distance:", total_distance)
