import numpy as np

# Load the history file
history = np.load("../logs/history.npy", allow_pickle=True).item()

# Print all keys
print("Available keys:", history.keys())

# Print values for each key
for key, value in history.items():
    print(f"\n{key}:")
    print(value)
