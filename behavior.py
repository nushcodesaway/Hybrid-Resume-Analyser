import random

def generate_behavior():
    return {
        "typing_speed": random.randint(20, 80),
        "time_spent": random.randint(5, 60),
        "copy_paste": random.randint(0, 10),
        "edits": random.randint(1, 15)
    }