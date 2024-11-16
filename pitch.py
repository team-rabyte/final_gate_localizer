import numpy as np

# Define the positions
P = np.array([0, 1])      # Drone's current position in XZ-plane (x_d, z_d)
M = np.array([1, 1])      # Target position in XZ-plane (x_m, z_m)

# Current heading vector (PX in XZ-plane)
PX = np.array([1, 0])     # Drone is heading along the positive X-axis

# Target direction vector (PM in XZ-plane)
PM = M - P

# Calculate the dot product and determinant
dot = np.dot(PX, PM)
det = PX[0] * PM[1] - PX[1] * PM[0]  # Determinant of 2D vectors

# Calculate the pitch angle using arctan2
pitch_angle_rad = np.arctan2(det, dot)  # Angle in radians
pitch_angle_deg = np.degrees(pitch_angle_rad)  # Convert to degrees

print(f"The pitch angle to face the target is: {pitch_angle_deg:.2f} degrees")
