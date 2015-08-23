import constants


def parse_file(filename):
    f = open(filename)
    f.readline()
    f.readline()
    f.readline()
    data = f.read().replace("\n", " ").replace("\t", " ").split(" ")
    return data[:-1]


def reduce_dimension(data):
    result = []
    for i in range(0, constants.IMAGE_HEIGHT, constants.REDUCE_HEIGHT):
        result.append([])
        for j in range(0, constants.IMAGE_WIDTH, constants.REDUCE_WIDTH):
            result[-1].append(average_gray(data, i, j))
    return result


def average_gray(data, i, j):
    result = 0
    for h in range(constants.REDUCE_HEIGHT):
        for w in range(constants.REDUCE_WIDTH):
            result += int(data[constants.IMAGE_WIDTH*(i+h)+j+w])
    return int(result/(constants.REDUCE_WIDTH*constants.REDUCE_HEIGHT))
