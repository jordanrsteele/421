def manhattan_distance(a, b):
    x = abs(CITY_DICT[a]['location'][0] - CITY_DICT[b]['location'][0])
    y = abs(CITY_DICT[a]['location'][1] - CITY_DICT[b]['location'][1])
    val = x + y
    return val

def euclidian_distance(a, b):
    x = abs(CITY_DICT[a]['location'][0] - CITY_DICT[b]['location'][0])
    y = abs(CITY_DICT[a]['location'][1] - CITY_DICT[b]['location'][1])

    euclidian_distance = (x**2 + y**2)**(1/2)
    return euclidian_distance
