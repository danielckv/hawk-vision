class SpeedCalculation:
    def __init__(self, speed_limit):
        self.speed_limit = speed_limit

    def calculate_speed(self, distance, time):
        return distance / time