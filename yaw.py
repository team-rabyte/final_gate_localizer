import numpy as np

# Define the positions
P = np.array([0, 0])      # Drone's current position
M = np.array([0, 1])      # Target position (center of square)

# Current heading vector (PX)
PX = np.array([1, 0])     # Drone is heading along the positive X-axis

# Target direction vector (PM)
PM = M - P

# Calculate the dot product and determinant
dot = np.dot(PX, PM)
det = PX[0] * PM[1] - PX[1] * PM[0]  # Determinant of 2D vectors

# Calculate the yaw angle using arctan2
yaw_angle_rad = np.arctan2(det, dot)  # Angle in radians
yaw_angle_deg = np.degrees(yaw_angle_rad)  # Convert to degrees

print(f"The yaw angle to face the target is: {yaw_angle_deg:.2f} degrees")
