"""
===================
=== SYNRL mk. 6 ===
===================

---
Author: Aman Bhargava
		University of Toronto
		Engineering Science
		Class of 2022
Date:   March 30, 2021
---

This is the sixth version of the synaptic reinforcement learning library. Primary changes:
	- [x] Arbitrary pathing for training result exports.
	- [x] __Add__ validation accuracy over time (every 10).
	- [x] __Add__ validation loss over time (every 10).
	- [x] __Add__ validation set & training set size.
	- [x] __Add__ final validation probability predictions, validation set true labels.
	- [x] __Add__ final training probability predictions, training set true labels.

This implements a traditional multi-layer perceptron system trained using a (seemingly novel)
synaptic reinforcement learning method. The goal is to create an easy-to-use fast library for
running training tests.
"""

using LinearAlgebra
using Random
using StatsFuns
using Statistics
using Plots
# using Plotly
using FileIO, JLD2
using NPZ
import JSON

using ProgressMeter

using InteractiveUtils

using UUIDs

"""
Global Constants
"""

const N_HIST = 2;
const W_RANGE = 0.1;
const SAVE_FOLDER = "../results/final/OCR"; # folder to save results into: SAVE_FOLDER/uuid.json
const NUM_CORES = 6;

"""
=== Neural Network Structs ===
 - Neuron
 - Layer
 - NeuralNetwork
 - Policy
"""

mutable struct Neuron
    w::Array{Float32, 1}
    actions::Array{Int8, 2}
end

mutable struct Layer
    neurons::Array{Neuron, 1}
	w_mat::Array{Float32, 2}
end

mutable struct NeuralNetwork
    layers::Array{Layer, 1}
    shape::Array{Int64, 1}
end

"""
Q[action][previous action][previous reward][next previous action][next previous reward][...]
is the expected value of the `action` given the previous data.

The number of `previous action`/`previous reward` pairs is given by `N_HIST`.

In order to index the current `action` and all previous `action/reward pairs`, we require
an array with dimensionality `N_HIST*2 + 1`.

In this version, each of `action` and `reward` can take on 3 values. Therefore, the
policy tensor is 3*3*3*3*...
"""
struct Policy
    Q::Array{Float32, N_HIST*2+1} # for each history round we have `actions` and `rewards`
end

"""
Utility Functions for Data Acquisition and Accuracy Measures
"""
function get_X_Y(notMNIST_path::String="../datasets/notMNIST.npz")
    dset = npzread(notMNIST_path)
    images = dset["images"];
    labels = dset["labels"];

    images = convert(Array{Float32,3}, images)
    typeof(images)
    # Reshaping X
    X = reshape(images, 18724, 784)
    X = transpose(X)
    size(X)
    # One-hot encoding the labels
    L = length(labels)
    Y = zeros(Int8, 10, L, )

    for i = 1:L
        Y[labels[i]+1, i] = 1
    end
    # Normalizing X
    X ./= 255;

    mask = convert(BitArray{1}, round.((bitrand(L) + bitrand(L))./2.5) );

    X_valid = X[:, mask];
    Y_valid = Y[:, mask];
    X_train = X[:, (!).(mask)];
    Y_train = Y[:, (!).(mask)];

    return X_train, Y_train, X_valid, Y_valid
end

function get_accuracy(network, x, y; activation::String="relu", verbose::Bool=false)

    y_hat = mass_forward(x, network, activation=activation);
	weh::Float32 = 10e-9;

	y_hat = sftmx(y_hat .+ weh)

	if verbose
		println("\n=== FROM GET_ACCURACY ===")
		println("ACC ACTIVATION: ",activation)
		println("y_hat: ")
		println(y_hat)
		println("TRUTH: ")
		println(y)
	end

    L = size(y)[2]
    num_correct = 0
    for i = 1:L
		if verbose
			println("Argmax of y_hat is: ",argmax(y_hat[:,i]))
			println("Truth is: ",argmax(y[:,i]))
		end
        if argmax(y_hat[:,i]) == argmax(y[:,i])
            num_correct += 1;
        end
    end

    return num_correct/L
end


function mass_forward(x::Array{Float32, 2}, network::NeuralNetwork; activation::String="relu")
    """
    Function for processing a batch of data through a neural network.
    x should be a matrix of column vectors where each column is a single piece of data.
    y will be of the same format with column vectors representing the output.
    """
	# println("x -> copy y")
	y::Array{Float32,2} = x;
	for i in 1:(length(network.layers)-1)
		# println("Add a bias: ")
		y = [1.0; y];
		# println("Matrix multiplication number ", i);
		y = network.layers[i].w_mat*y
		if activation == "tanh"
			# println("Activation (tanh): ")
			y = tanh.(y)
		else
			# println("Activation (ReLU): ")
			y = max.(zero(y), y);
		end
	end

	# println("Add a bias: ")
	y = [1; y];
	# println("Final Matrix Multiplication: ")
	y = network.layers[end].w_mat*y
	# println("FINAL TIME: ")

	return y
end


"""
=== Generate Functions ===
"""

function generate_neuron(dim::Int64; zero_bias::Bool=true)
    # Generate history matrix

    actions = zeros(dim, N_HIST+1).+2 # For DIM synapses we store their actions.
                                    # Starting with N_HIST actions set to 0.
    actions = convert(Array{Int8, 2},actions)

    # Generate weight vector
    w = rand(Float32, dim);
    w*=2;
    w =w.-1;
    w*=W_RANGE;

    # Set bias to 0 by default
    if zero_bias
        w[1] = 0;
    end
    neur = Neuron(w,actions);
    return neur;
end


function generate_layer(num_neurons::Int64, dim_neuron::Int64; zero_bias::Bool=true)
    """
    num_neurons: Number of neurons in the layer
    dim_neuron:  Number of neurons in the previous layer (including bias).
    """
    neuron_array = Array{Neuron,1}(undef, num_neurons);
	mat = zeros(num_neurons, dim_neuron)

    for i = 1:num_neurons
        neuron_array[i] = generate_neuron(dim_neuron, zero_bias=zero_bias);
		mat[i,:] = neuron_array[i].w;
    end

    return Layer(neuron_array, mat);
end

function generate_network(dimensions::Array{Int64, 1}; zero_bias::Bool=true)
    """
    dimensions:  List of layer dimensionality (starting with input layer and ending with output layer).
                 Must be a minimum of 2-long.
    """
    if(length(dimensions) < 2)
        println("dimensions array too short in `generate_network`");
        return false;
    end

    layer_array = Array{Layer,1}(undef, length(dimensions)-1)

    for i = 1:(length(dimensions)-1)
        layer_array[i] = generate_layer(dimensions[i+1], dimensions[i]+1, zero_bias=zero_bias);
    end

    net = NeuralNetwork(layer_array, dimensions);

    return net
end

function new_policy()
    size = convert(Array{Int,1}, ones(N_HIST*2+1)*3)
    size_tuple = tuple(size...)
    pol = Policy(zeros( size_tuple ))
    return pol
end




"""
Loss Functions and Related
"""
function sftmx(X::Array{Float32})
	"""
	Softmax function for column-output vectors concatenated into matrices.
	Works for 1-D outputs as sigmoid function.
	"""
	if(size(X)[1] == 1) # we just apply sigmoid
		X = logistic.(X);
	else
		for i = 1:size(X)[2]
			X[:,i] = X[:,i] .- maximum(X[:,i])
			X[:,i] = exp.(X[:,i])
			X[:,i] = X[:,i] / sum(X[:,i])
		end
	end
	return X
end

function lp_loss(net::NeuralNetwork, x::Array{Float32,2}, truth::Array{Int8,2}; softmax::Bool=true,
				 activation::String="relu", verbose::Bool=false, p::Float64=5.)
	"""
	Loss function based on the ℓₚ norm family.

	After a softmax function is applied to the output, the LP norm is calculated
	between ||1-y_hat[c]|| where c is the correct class.

	The loss for training example n is therefore ||1-y_hat_n[c]||.
		If y_hat_n[c] == 1, we are correct. Therefore, loss is ||1-1|| = 0
		If y_hat_n[c] == 0, we are thoroughly incorrect. Therefore, loss is ||1-0|| = 1

	The lp-norm is there to penalize incorrect values very highly if they are far from 1.

	The sum over all examples represented in (x, truth) is outputted. Probably divided
	by N for safety.
	"""
	pred = mass_forward(x, net, activation=activation);
	weh::Float32 = 10e-9;

	pred = sftmx(pred .+ weh)

	N = size(x)[2]

	@assert N == size(pred)[2]

	sum_loss = 0.0

	for i = 1:N
		gnd_trth = argmax(truth[:,i])
		difference = 1-pred[gnd_trth,i]
		if p == 0.0
			sum_loss += max(2*difference-1,0)
		else
			sum_loss += difference^p
		end

	end

	return sum_loss / N
end

function CE_loss(net::NeuralNetwork, x::Array{Float32,2}, truth::Array{Int8,2}; softmax::Bool=true,
			  activation::String="relu", verbose::Bool=false, p::Float64=5.)
	"""
	Cross-entropy loss function for matrix comparison.
	By default, `softmax` is set to true so each column of the
	output of `mass_forward(x, net)` is put through a softmax function
	before comparison via cross entropy loss.
	"""
	pred = mass_forward(x, net, activation=activation);
	weh::Float32 = 10e-9;

	pred = sftmx(pred .+ weh)

	if verbose
		println("\n=== FROM CE_LOSS ===")
		println("CE_LOSS ACTIVATION: ",activation)
		println("PRED: ")
		println(pred)
		println("TRUTH: ")
		println(truth)
	end

	return (-1/size(x)[2])*sum(truth.*log.(pred .+ 10e-9));
end

function loss(net::NeuralNetwork, x::Array{Float32,2}, truth::Array{Int8,2}; softmax::Bool=true,
			  activation::String="relu", verbose::Bool=false, p::Float64=5.)

	# return lp_loss(net=net, x=x, truth=truth; softmax=softmax, activation=activation, verbose=verbose, p=p)
	return CE_loss(net, x, truth; softmax=softmax, activation=activation, verbose=verbose, p=p)
end

function take_synaptic_step(net::NeuralNetwork, rewards::Array{Float64, 1}, pol::Policy,
							exploration_probability, ϵ, α; use_bias=false)
	"""
	Function that uses `policy` to take one step of size ϵ for each neuron in `net`.
	"""
	# for (l_idx, layer) in enumerate(net.layers)
	for l_idx in 1:length(net.layers)
		# for (n_idx, neuron) in enumerate(layer.neurons)
		# layer = net.layers[l_idx];
		Threads.@threads for n_idx in 1:length(net.layers[l_idx].neurons)
			# neuron = net.layers[l_idx].neurons[n_idx];

			net.layers[l_idx].neurons[n_idx].actions = circshift(net.layers[l_idx].neurons[n_idx].actions, (0, -1)) # Shifting each row of the neuron's history
															   # one to the left
			net.layers[l_idx].neurons[n_idx].actions[:,end] .= 0 # Setting the element that looped around to the end to zero
									   # (this is the element that will be populated here)
			# Iterating through each synapse and taking an action (inc/dec/same).
			for j = 1:length(net.layers[l_idx].neurons[n_idx].w)
				# The address for synapse j consists of the last N_HIST actions concatenated
				# with the last N_HIST rewards.
				# Example: For N_HIST = 2, addr = [action_{t-1}, action_{t-2}, reward_{t-1}, reward_t-2]
				# Since actions and rewards take on values of either 1, 2, or 3 (indicating
				# decrease, stay the same, or increase), one can use them to index an array in
				# Julia (a 1-indexed language).
#                     addr = [actions[j, end-N_HIST:end-1]; rewards[end-N_HIST:end-1]]
				addr = [net.layers[l_idx].neurons[n_idx].actions[j, end-N_HIST:end-1]; rewards[end-N_HIST:end-1]] # TODO: Test this.
				addr = convert(Array{Int64,1}, addr)

				a = 0; # Action taken by agent
				if rand() < exploration_probability
					a = rand([1, 2, 3]); # Chose a random move
				else
					if(j == 1 && use_bias) # If we are on the bias term...
						a = argmax(bias_pol.Q[:,addr...]); # Choose the "optimal" move given the
														  # Past actions and rewards
					else
						a = argmax(pol.Q[:,addr...]); # Choose the "optimal" move given the
													  # Past actions and rewards
					end
				end

				# We have now chosen our action.
				net.layers[l_idx].neurons[n_idx].actions[j,end] = a;

				# Taking the action by adjusting the synapse weight accordingly.
				if j == 1
					net.layers[l_idx].neurons[n_idx].w[j] += (a-2)*ϵ*2; # TEST: Using smaller learning rate for bias term
					net.layers[l_idx].w_mat[n_idx, j] += (a-2)*ϵ*2;
				else
					net.layers[l_idx].neurons[n_idx].w[j] += (a-2)*ϵ;
					net.layers[l_idx].w_mat[n_idx, j] += (a-2)*ϵ;
				end
			end # iterating through each neuron weight vector entry
		end # Iterating through each neuron in the layer
	end
end



function train_network(x, truth, dim::Int64, iters::Int64, explore_prob,
                       net_shape::Array{Int64, 1}; train_policy::Bool=true, pol::Any=false,
                       bias_pol::Any=false, use_bias::Bool=false, batch_size::Int64=300,
					   batch_interval::Int64=30, ϵ = 0.001, α=0.01, γ  = 0.9, activation = "relu",
					   p::Float64=5., acc_sample_period::Int64=10, switch_iteration::Int64=-1, switch_epsilon::Float64=-1.,
					   x_valid, y_valid)
    """
    Function to train a new neural network of shape `net_shape` to map column vectors of `x`
	accurately to one-hot vectors in `truth`.

    Inputs
    ---
	`x`				  Input data (set of column vectors). Vector dimensionality must match
	                  first entry in `net_shape`.
	`truth`			  Data labels (set of column vectors, one-hot for classification). Vector
	  				  dimensionality must match last entry in `net_shape`.
    `dim`             Dimension of the input data.
    `iters`           Number of iterations (epochs) to train the network for.
    `explore_prob`    Probability of a neuron taking a random training step (~50%).
    `net_shape`       Size of each network layer (including input).
                      Note that the first entry should be equal to `dim` and
                      the first and last entries should match the first and last layer
                      sizes of `target`.
    `train_policy`    Boolean for if we want to use Q-learning to keep improving the policy.
    `pol`             Provided policy. If the user wants a new policy to be generated, set
                      it's value to `false`.
    `bias_pol`        Provided policy for training the bias. See input rules for `pol`.
    `use_bias`        Boolean for if we shoudl actually make use of a separate bias policy
                      (either the provided one or one that we generate if `bias_pol` is
                      set to false).

	`batch_size`	  Number of training samples in each batch.
	`batch_interval`  Number of iterations for which a given batch is trained upon before
					  the batch is changed out for a new one.

    Outputs
    ---
    actions, rewards, losses, percep, pol, bias_pol
    """

    num_iterations = iters;
    exploration_probability = explore_prob;
    net = generate_network(net_shape);

	# Initializing the mini-batching mask bitarray.
	batch_mask = falses(size(x,2))
	batch_mask[1:batch_size] .= true
	batch_mask = shuffle(batch_mask)

	println("Batch mask sum: ", sum(batch_mask))

    if pol == false || typeof(pol) != Policy
        pol = new_policy();
        bias_pol = new_policy();
        train_policy = true;
    end

    rewards = zeros(N_HIST).+2 # Rewards history is instantiated with N_HIST zeros.
    losses = zeros(N_HIST).+loss(net, x[:,batch_mask], truth[:,batch_mask], p=p)
	valid_losses = []
	accuracies = []
	valid_accuracies = []

    println("Starting training loop");
	# for i = 1:num_iterations
	@showprogress 1 "Computing..." for i = 1:num_iterations
		if(i % batch_interval == 0) # Switching batch mask every `batch_interval` iterations.
			batch_mask = shuffle(batch_mask)
		end

		if i == switch_iteration # switching epsilon to `switch_epsilon` upon iteration == switch_iteration
			if switch_epsilon != -1
				ϵ = switch_epsilon;
			end
		end

        append!(rewards, 0) # appending a zero to the end of the reward array to
                            # record the reward in this iteration.
		take_synaptic_step(net, rewards, pol, exploration_probability, ϵ, α)
        # Calculating reward for this iteration through all synapses.

        new_loss = loss(net, x[:, batch_mask], truth[:, batch_mask], activation=activation, p=p);
		if i % acc_sample_period == 0
			# Calculating training accuracy across entire training set.
			append!(accuracies, get_accuracy(net, x, truth, activation=activation) );
			# Calculating validation accuracy across entire validation set.
			append!(valid_accuracies, get_accuracy(net, x_valid, y_valid, activation=activation) );
			# Calculating validation loss across entire validation set.
			append!(valid_losses, loss(net, x_valid, y_valid, activation=activation, p=p) )

			if(i > 25_000 && accuracies[end] < 0.2)
				println("\n\n\nITERATION 25,000 WITH ACCURACY < 0.2");
				println("INDICATIVE OF FAILED TRAINING. BREAKING FROM LOOP.\n\n\n");
				break;
			end
		end

        R = 0; # R is the reward for is previous action.
        # Reward is dictated by whether or not loss increased or decreased overall.
        if(new_loss < losses[end])
            R = 3;
        else
            R = 1;
        end

        # println("Old loss: ", losses[end])
        # println("New loss: ", new_loss)
        # println("Reward: ", R)

        rewards[end] = R;
        append!(losses, new_loss)

        # Applying update rule for Q values on each synapse
		# TODO: Make this a dynamic program (store q-value updates concurrently, apply them all at once quickly at the end).
        if train_policy
            for layer in net.layers
                for neuron in layer.neurons
                    for j = 1:length(neuron.w)
                        addr_old = [neuron.actions[j, end-N_HIST:end-1]; rewards[end-N_HIST:end-1]]
                        addr_new = [neuron.actions[j, end-N_HIST+1:end]; rewards[end-N_HIST+1:end]]

                        addr_old = convert(Array{Int64,1}, addr_old)
                        addr_new = convert(Array{Int64,1}, addr_new)

                        if j == 1 && use_bias # control flow for BIAS term
                            max_future_term = γ*maximum(bias_pol.Q[:,addr_new...]);
        #                     max_future_term = maximum(bias_pol.Q[:,addr_new...]); # TEST: Using gamma = 1 for bias

                            cur_term = bias_pol.Q[ convert(Int64, neuron.actions[j,end]) , addr_old...];

                            bias_pol.Q[ convert(Int64, neuron.actions[j,end]) , addr_old...] =
                            cur_term + α*( (R-2) + max_future_term - cur_term);
                        else # control flow for NON-BIAS terms
                            max_future_term = γ*maximum(pol.Q[:,addr_new...]);

                            cur_term = pol.Q[ convert(Int64, neuron.actions[j,end]) , addr_old...];

                            pol.Q[ convert(Int64, neuron.actions[j,end]) , addr_old...] =
                            cur_term + α*( (R-2) + max_future_term - cur_term);
                        end

                    end
                end
            end
		end
    end

    # Returning all actions taken, all rewards, all losses
    # And the final perceptron and the final perceptron learning policy.
    return rewards, losses, net, pol, bias_pol, accuracies, valid_losses, valid_accuracies
end

function run_experiment(verbose::Bool; net_size::Array{Int64,1}=[784,1000,10], iterations::Int64=20000,
						activation::String="relu", explore_prob=0.2, policy_path::String="golden_pol2cy.jld2",
						train_policy::Bool=false, batch_size::Int64=300, batch_interval::Int64=30,
						ϵ=0.001, α=0.01, γ=0.9, p::Float64=5., acc_sample_period::Int64=10,
						switch_iteration::Int64=-1, switch_epsilon::Float64=-1.)
	println("JULIA NUM THREADS: ",Threads.nthreads())
	if(verbose)
		println("Setting number of BLAS threads...")
	end
	BLAS.set_num_threads(NUM_CORES);
	if verbose
		println("Getting Dataset...");
	end
	X_train, Y_train, X_valid, Y_valid = get_X_Y();

	# Get the policy from disk.
	policy::Policy = new_policy();

	if policy_path != "None"
		if verbose
		    println("Getting policy...")
		end
	    policy = FileIO.load(policy_path, "pol");
	    if verbose
	        println("Policy loaded\n")
	    end
	end

	println("Size of X_train: ", size(X_train))

	# x = X_train[:,1:10]
	# y = Y_train[:,1:10]

	x = X_train
	y = Y_train

	@time rewards, losses, net, pol, bias_pol, accuracies, valid_losses, valid_accuracies = train_network(
														x, y, 784, iterations, explore_prob, net_size, train_policy=train_policy,
														pol=policy, batch_size=batch_size, batch_interval=batch_interval,
														ϵ=ϵ, α=α, γ=γ, activation=activation, p=p, acc_sample_period=acc_sample_period,
														switch_iteration=switch_iteration, switch_epsilon=switch_epsilon,
														x_valid=X_valid, y_valid=Y_valid
													)

	train_acc = get_accuracy(net, x, y, activation=activation);
	valid_acc = get_accuracy(net, X_valid, Y_valid, activation=activation);
	train_loss = loss(net, x, y, activation=activation, p=p);
	valid_loss = loss(net, X_valid, Y_valid, activation=activation, p=p);

	# Estimated class probabilities for the entire training set.
	train_probabilities = mass_forward(X_train, net, activation=activation);
	weh::Float32 = 10e-9;
	train_probabilities = sftmx(train_probabilities .+ weh);

	valid_probabilities = mass_forward(X_valid, net, activation=activation); # Estimated class probabilities for the entire validation set.
	valid_probabilities = sftmx(valid_probabilities .+ weh);

	if verbose
	    println("=== RESULTS ===")
	    println("Accuracy on training set: \t\t", train_acc);
	    println("Accuracy on validation set: \t\t", valid_acc);
	    println("Final training loss: \t\t\t", train_loss);
	    println("Final validation loss: \t\t\t", valid_loss);

	    # display(plot(losses, title="Losses", lw = 3));
		# display(plot(accuracies, title="Accuracies", lw = 3));
	end

	hidden_units = 0
	if length(net_size) > 2
		hidden_units = net_size[2]
	end
	rng = MersenneTwister(1234);
	savename = string(SAVE_FOLDER,"/",uuid1(rng),".json");

	output_dict = Dict{String, Any}(
	    "W_RANGE" => W_RANGE,
	    "N_HIST" => N_HIST,
	    "epsilon" => ϵ,
	    "alpha" => α,
	    "gamma" => γ,
		"activation" => activation,
	    "iters" => iterations,
	    "hidden_units" => hidden_units,
	    "policy_path" => policy_path,
	    "train_policy" => train_policy,
	    "exp_prob" => explore_prob,
	    "savename" => savename,
	    "batch_size" => batch_size,
	    "batch_interval" => batch_interval,
	    "losses" => losses,
		"accuracies" => accuracies,
		"acc_sample_period" => acc_sample_period,
	    "train_loss" => train_loss,
	    "valid_loss" => valid_loss,
	    "train_acc" => train_acc,
	    "valid_acc" => valid_acc,
		"switch_iteration" => switch_iteration,
		"switch_epsilon" => switch_epsilon,
		"valid_losses" => valid_losses,
		"valid_accuracies" => valid_accuracies,
		"train_probabilities" => train_probabilities,
		"valid_probabilities" => valid_probabilities,
		"train_labels" => Y_train,
		"valid_labels" => Y_valid
	);

	output_string = JSON.json(output_dict)

	open(savename, "w") do f
	    write(f, output_string)
	end
end


function main()
	"""
	Function to (hopefully) achieve the final results for these experiments.
	Starting with the least computationally intensive simulations, we will
	retrieve all of the data we need to create our final plots.

	Final experiments to run:
	 - https://www.notion.so/Experiment-Finalization-dff7b1ab955e497a89429ede017d20b6
	 - Batch size: 5000
	 - Batch interval: 5000
	 - Epsilon: 0.0001
	 - Exploration probability: 0.4
	 - Iterations: 100000 (maybe more later...)

	Running the experiments 5 times for each (0, 10, 32) hidden layers.
	"""
	# ITERS = 2000
	# ITERS = 200_000
	# ITERS = 500
	ITERS = 500_000

	BATCH_SIZE = 5000 # Seems roughly optimal
	BATCH_INTERVAL = 5000 # Maybe should be longer?
	EPSILON = 0.0001 # This is the way.
	EXP_PROB = 0.1 # Seems like anything is OK between 0.1-0.4?
	SWITCH_ITERATION = -1

	SWITCH_ITERATION_10 = -1
	SWITCH_ITERATION_32 = 100_000

	SWITCH_EPSILON = 0.00005

	HIDDEN_LAYERS = [32]

	for ex = 1:9 # Repeating trials 10 times...
		for HIDDEN_LAYER in HIDDEN_LAYERS
			NET_SIZE = [784, 10]
			if HIDDEN_LAYER != 0
				NET_SIZE = [784, HIDDEN_LAYER, 10]
			end

			if HIDDEN_LAYER == 32
				SWITCH_ITERATION = SWITCH_ITERATION_32
			else
				SWITCH_ITERATION = SWITCH_ITERATION_10
			end

			println("\n\n=== ",ex,": BEGINNING EXPERIMENT"," ===")
			println("\tHidden Layers: ",HIDDEN_LAYER)
			println("\tSwitch iteration: ", SWITCH_ITERATION)
			println("\tSwitch epsilon: ", SWITCH_EPSILON)
			println("\tExperiment Round: ",ex,"/10\n")


			@fastmath run_experiment(true, iterations=ITERS, batch_size=BATCH_SIZE, batch_interval=BATCH_INTERVAL,
									 net_size=NET_SIZE, activation="relu", ϵ=EPSILON, explore_prob=EXP_PROB,
									 train_policy=false, p=0., acc_sample_period=10, switch_iteration=SWITCH_ITERATION, switch_epsilon=SWITCH_EPSILON);
		end
	end
end


main()
