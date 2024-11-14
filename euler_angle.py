import numpy as np

def calculate_euler_angles(A, B, C, D, P):
    """
    Calculates the Euler angles (yaw, pitch, roll) required for a drone at point P
    to align with a square frame defined by points A, B, C, and D.
    
    Parameters:
    A, B, C, D : ndarray
        Coordinates of the four corners of the square frame.
    P : ndarray
        Position of the drone.
        
    Returns:
    tuple
        Yaw, Pitch, and Roll angles in degrees.
    """
    # Calculate the center of the square (M) as the average of the four corners
    M = (A + B + C + D) / 4

    # Calculate vector from P to M
    delta_x, delta_y, delta_z = M - P

    # Calculate Yaw angle
    yaw_angle = np.degrees(np.arctan2(delta_y, delta_x))

    # Calculate Pitch angle
    pitch_angle = np.degrees(np.arctan2(delta_z, np.sqrt(delta_x**2 + delta_y**2)))

    # Calculate the normal vector of the square's plane
    AB = B - A
    AC = C - A
    normal_vector = np.cross(AB, AC)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize

    # Assume drone's right vector is along the X-axis (for simplicity)
    drone_right_vector = np.array([1, 0, 0])

    # Calculate Roll angle
    roll_angle = np.degrees(np.arccos(np.dot(drone_right_vector, normal_vector)))

    return yaw_angle, pitch_angle, roll_angle

# Example usage
A = np.array([8, 10, 2])
B = np.array([12, 10, 2])
C = np.array([12, 10, 0])
D = np.array([8, 10, 0])
P = np.array([1, 10, 1])

print(calculate_euler_angles(A, B, C, D, P))
