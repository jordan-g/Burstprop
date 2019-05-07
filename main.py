import argparse
import numpy as np

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-model", help='Model to use.', default='sigmse')
parser.add_argument("-data_directory", help='Directory in which to save data.')
parser.add_argument("-dataset", type=str, help="Which dataset to train on ('mnist' or 'cifar10').", default="mnist")
parser.add_argument("-num_epochs", type=int, help="Number of epochs.", default=10)
parser.add_argument("-num_hidden_layers", type=int, help="Number of hidden layers.", default=2)
parser.add_argument("-batch_size", type=int, help="Batch size.", default=1)
parser.add_argument("-test_frequency", type=int, help="Frequency (ie. every ___ batches) at which to save network state & get test error.", default=1000)
parser.add_argument("-num_hidden_units", help="Number of units in each hidden layer.", type=lambda s: [int(item) for item in s.split(',')], default=[500, 500])
parser.add_argument("-Z_range", help="Range of uniform distribution used for initial recurrent weights.", type=lambda s: [float(item) for item in s.split(',')], default=[0.01, 0.01])
parser.add_argument("-Y_range", help="Range of uniform distribution used for initial feedback weights.", type=lambda s: [float(item) for item in s.split(',')], default=[1.0, 1.0])
parser.add_argument("-forward_learning_rates", help="Feedforward learning rates.", type=lambda s: [float(item) for item in s.split(',')], default=[0.1, 0.1, 0.1])
parser.add_argument("-recurrent_learning_rates", help="Recurrent learning rates.", type=lambda s: [float(item) for item in s.split(',')], default=[0.0001, 0.0001])
parser.add_argument("-gamma", type=float, help="Output layer burst probability in the absence of a target.", default=0.2)
parser.add_argument("-beta", type=float, help="Slope of the burst probability function.", default=1.0)
parser.add_argument("-symmetric_weights", type=lambda x: (str(x).lower() == 'true'), help="Whether to use symmetric weights.", default=False)
parser.add_argument("-same_sign_weights", type=lambda x: (str(x).lower() == 'true'), help="Whether to use feedback weights that are the same sign as feedforward weights.", default=True)
parser.add_argument("-use_recurrent_input", type=bool, help="Whether to use recurrent input at hidden layers.", default=True)
parser.add_argument("-momentum", type=float, help="Momentum", default=0.8)
parser.add_argument("-weight_decay", type=float, help="Weight decay", default=0.0)
parser.add_argument("-heterosyn_plasticity", type=float, help="Heterosynaptic plasticity", default=0.)
parser.add_argument("-use_validation", type=lambda x: (str(x).lower() == 'true'), help="Whether to use the validation set.", default=False)
parser.add_argument("-use_backprop", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("-p_0", type=float, help="Baseline probability for all layers", default=0.5)

args = parser.parse_args()

# create a hyperparameters dictionary
hyperparameters = {
	"model":                    args.model,
	"data_directory":           args.data_directory,
	"dataset":                  args.dataset,
	"num_epochs":               args.num_epochs,
	"num_hidden_layers":        args.num_hidden_layers,
	"batch_size":               args.batch_size,
	"test_frequency":           args.test_frequency,
	"num_hidden_units":         args.num_hidden_units,
	"Z_range":                  args.Z_range,
	"Y_range":                  args.Y_range,
	"forward_learning_rates":   args.forward_learning_rates,
	"recurrent_learning_rates": args.recurrent_learning_rates,
	"gamma":                    args.gamma,
	"beta":                     args.beta,
	"symmetric_weights":        args.symmetric_weights,
	"same_sign_weights":        args.same_sign_weights,
	"use_recurrent_input":      args.use_recurrent_input,
	"momentum":                 args.momentum,
	"weight_decay":             args.weight_decay,
	"use_validation":           args.use_validation,
	"use_backprop":             args.use_backprop,
	"p_0":                      args.p_0,
	"heterosyn_plasticity":		args.heterosyn_plasticity
}

def initialize_network(hyperparameters):
	global net
	if args.model == 'sigmse':
		import sigmse.network as net
	elif args.model == 'sigmse_conv':
		import sigmse.conv_network as net
	elif args.model == 'sigmse_time':
		import sigmse.time_network as net
	elif args.model == 'expcrossent':
		import expcrossent.network as net

	parameters, state, gradients = net.initialize(hyperparameters)

	return parameters, state, gradients

def train(x_train, d_train, x_test, d_test, parameters, state, gradients, hyperparameters):
	num_batches = int(d_train.shape[1]/hyperparameters["batch_size"])

	train_cost  = np.zeros(hyperparameters["num_epochs"]*num_batches)
	train_error = np.zeros(hyperparameters["num_epochs"]*num_batches)
	test_error  = np.zeros(hyperparameters["num_epochs"]*num_batches+1)

	test_error[0] = net.test(x_test, d_test, parameters, hyperparameters)

	print("Initial {} error: {:.2f}%.".format("validation" if hyperparameters["use_validation"] else "test", test_error[0]))

	example_indices = np.arange(x_train.shape[1])

	for epoch_num in range(hyperparameters["num_epochs"]):
		for batch_num in range(num_batches):
			batch_example_indices = example_indices[batch_num*hyperparameters["batch_size"]:(batch_num+1)*hyperparameters["batch_size"]]

			total_batch_num = epoch_num*num_batches + batch_num

			x = x_train[:, batch_example_indices]
			d = d_train[:, batch_example_indices]

			state, gradients, train_cost[total_batch_num], train_error[total_batch_num] = net.update(x, d, parameters, state, gradients, hyperparameters)

			parameters = net.update_weights(parameters, state, gradients, hyperparameters)

			if (batch_num+1) % hyperparameters["test_frequency"] == 0:
				test_error[total_batch_num+1] = net.test(x_test, d_test, parameters, hyperparameters)

				print("Epoch {}, batch {}/{}. {} error: {:.2f}%. Train cost: {}.".format(epoch_num+1, batch_num+1, num_batches, "Validation" if hyperparameters["use_validation"] else "Test", test_error[total_batch_num+1], np.mean(train_cost[total_batch_num+1-hyperparameters["test_frequency"]:total_batch_num+1])))

	return train_cost, train_error, test_error

if __name__ == "__main__":
	# initialize the network
	parameters, state, gradients = initialize_network(hyperparameters)
	
	# load the dataset
	x_train, d_train, x_test, d_test = net.load_dataset(hyperparameters["dataset"], hyperparameters["use_validation"])

	# train the network
	train_cost, train_error, test_error = train(x_train, d_train, x_test, d_test, parameters, state, gradients,  hyperparameters)