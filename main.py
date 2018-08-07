def initialize_network(hyperparameters):
	# for each layer
	initialize_params()

	return parameters

def train(x_train, y_train, x_validation, y_validation, parameters, hyperparameters):

	compute_prediction()
	comptue_gradients()

	# update weights

	return train_cost, train_error, validation_error

def compute_error(x, y, parameters):

	prediction = compute_prediction(x, parameters)

	return error

def compute_cost(pediction, y):

	return cost

def initialize_params(n_in, n_out, pdf='normal'):

	return W, bias

def __main__():
	initialize_network()
	train()