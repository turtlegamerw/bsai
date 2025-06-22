#superstick center
superstick_x, superstick_y = 1660, 800

# Directions (dx, dy offset) aiming super
directions = {
    "up": (0, -200),
    "down": (0, 200),
    "left": (-200, 0),
    "right": (200, 0),
    "up_right": (150, -150),
    "down_left": (-150, 150),
    "up_left": (-150, -150),       
    "down_right": (150, 150)       
}
#temporary because no ai
def aimsuper(randomnum):
    # Map random number to direction name
    direction_keys = [
        "up",          # 1
        "down",        # 2
        "left",        # 3
        "right",       # 4
        "up_right",    # 5
        "down_left",   # 6
        "up_left",     # 7
        "down_right"   # 8
    ]
    if 1 <= randomnum <= len(direction_keys):
        direction = direction_keys[randomnum - 1]
        dx, dy = directions[direction]
        end_x =superstick_x + dx
        end_y =superstick_y + dy
        return superstick_x, superstick_y, end_x, end_y
    else:
        raise ValueError("Invalid randomnum. Must be between 1 and 8.")