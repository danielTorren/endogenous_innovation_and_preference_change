class VehicleUser:
    def __init__(self, user_id, chi, gamma, beta, nu,  origin, vehicle):
        self.user_id = user_id  # Unique identifier for each user
        self.chi = chi  # Threshold for EV consideration (openness to innovation)
        self.gamma = gamma  # Environmental sensitivity
        self.beta = beta  # Cost sensitivity
        self.nu = nu # Time senstivity
        self.origin = origin  # User's origin (urban or rural), impacts public transport availability
        self.vehicle = vehicle
        self.current_vehicle_type = self.vehicle.type  # Current vehicle type (e.g., ICE,EV,Urban public transport or Rural public transport)
