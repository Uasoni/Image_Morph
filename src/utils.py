def get_coords(index , width):
    return index % width, index // width

def get_aspect_ratio(width, height):
    return width / height