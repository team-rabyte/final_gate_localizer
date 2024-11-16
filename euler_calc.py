import numpy as np

def calculate_yaw(P, M):
    """Calculate yaw in the XY-plane."""
    PX = np.array([1, 0])  # Drone is heading along the positive X-axis
    PM = M - P
    dot = np.dot(PX, PM)
    det = PX[0] * PM[1] - PX[1] * PM[0]
    yaw_angle_rad = np.arctan2(det, dot)
    return np.degrees(yaw_angle_rad)

def calculate_pitch(P, M):
    """Calculate pitch in the XZ-plane."""
    PX = np.array([1, 0])  # Drone is heading along the positive X-axis
    PM = M - P
    dot = np.dot(PX, PM)
    det = PX[0] * PM[1] - PX[1] * PM[0]
    pitch_angle_rad = np.arctan2(det, dot)
    return np.degrees(pitch_angle_rad)

def euler_angles(square_coords):
    # Calculate the center of the square
    x_coords = [coord[0] for coord in square_coords]
    y_coords = [coord[1] for coord in square_coords]
    center = np.array([np.mean(x_coords), np.mean(y_coords)])  # Center of the square
    
    # Drone is at the origin in both XY-plane and XZ-plane
    drone_xy = np.array([0, 0])
    drone_xz = np.array([0, 0])
    
    # Calculate yaw (XY-plane) and pitch (XZ-plane)
    yaw = calculate_yaw(drone_xy, center)
    pitch = calculate_pitch(np.array([0, 1]), np.array([center[0], 1]))  # Assume Z = 1 for pitch
    
    return yaw, pitch
