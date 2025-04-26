import sys, os


# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from library_data_2023 import telemetry, laps, events, weather, results, fastestlaps


def straights_speed():
    return False
def corners():
    return False
def braking_efficiency():
    return False
def tyre_deg():
    return False
def combined():
    return False