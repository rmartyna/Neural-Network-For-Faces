import network
import constants

# Example:
# Train networks for each type: facial expression, face direction, with or without sunglasses
print("Test sunglasses:")
net_sunglasses = network.Network(constants.INPUT_SIZE, 16, constants.OUTPUT_SUNGLASSES, 1)
net_sunglasses.train(constants.TEST_SUNGLASSES)
net_sunglasses.test(constants.TEST_SUNGLASSES)
print("Test expression:")
net_expression = network.Network(constants.INPUT_SIZE, 16, constants.OUTPUT_EXPRESSION, 1)
net_expression.train(constants.TEST_EXPRESSION)
net_expression.test(constants.TEST_EXPRESSION)
print("Test direction:")
net_direction = network.Network(constants.INPUT_SIZE, 16, constants.OUTPUT_DIRECTION, 1)
net_direction.train(constants.TEST_DIRECTION)
net_direction.test(constants.TEST_DIRECTION)
