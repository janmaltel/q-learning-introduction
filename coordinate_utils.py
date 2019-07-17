def position_to_image_coordinates(position, num_cols):
    # % is the "modulo operator"
    x = position % num_cols
    # "/" is an integer division
    y = position // num_cols
    return x, x + 1, y, y + 1

def one_d_to_two_d(position, num_cols):
    # % is the "modulo operator"
    x = position % num_cols
    # "/" is an integer division
    y = position // num_cols
    return x, y

def direction_to_x_y(direction):
    if direction == 0:
        x, y = (0, 1)
    elif direction == 1:
        x, y = (1, 0)
    elif direction == 2:
        x, y = (0, -1)
    elif direction == 3:
        x, y = (-1, 0)
    return (x, y)