from datetime import datetime


def create_time_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
