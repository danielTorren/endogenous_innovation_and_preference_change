import numpy as np

class VehicleUser:
    def __init__(self, user_id):
        self.user_id = user_id  # Unique identifier for each user
        self.vehicle = None  # Initial vehicle decision
