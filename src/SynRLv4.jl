"""
===================
=== SYNRL mk. 4 ===
===================

---
Author: Aman Bhargava
		University of Toronto
		Engineering Science
		Class of 2022
Date:   March 30, 2021
---

This is the fourth version of the synaptic reinforcement learning library. Primary changes:
 - [ ] Functionalizing everything.
 - [ ] Progressive speed testing.
     - [ ] Using `@time`
 - [ ] Separation of alloaction + algorithmic sections of code.
 - [ ] Ensuring strong typing for struct fields.

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

"""
Global Constants
"""

const N_HIST = 2;

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

function generate_neuron(dim::Int64; zero_bias::Bool=true, W_RANGE::Float32=2.0)
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
		weh::Float32 = 2.0
        neuron_array[i] = generate_neuron(dim_neuron, zero_bias=zero_bias, W_RANGE=weh);
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

function lp_loss(net::NeuralNetwork, x::Array{Float32,2}, truth::Array{Int8,2}, softmax::Bool=true;
				 activation::String="relu", verbose::Bool=false, p::Float64=5.)
	"""
	Loss function based on the ℓₚ norm family.

	After a softmax function is applied to the output, the LP norm is calculated
	between ||1-y_hat[c]|| where c is the correct class.

	The sum over all examples represented in (x, truth) is outputted. Probably divided
	by N for safety.
	"""
	pred = mass_forward(x, net, activation=activation);
	weh::Float32 = 10e-9;

	pred = sftmx(pred .+ weh)

end

function CE_loss(net::NeuralNetwork, x::Array{Float32,2}, truth::Array{Int8,2}, softmax::Bool=true;
				 activation::String="relu", verbose::Bool=false)
	"""
	Cross-entropy loss function for matrix comparison.
	By default, `softmax` is set to true so each column of the
	output of `mass_forward(x, net)` is put through a softmax function
	before comparison via cross entropy loss.
	"""
	@assert false; # CE loss should probably never be called again, honestly.

	## TEMPORARILY REPLACING WITH NEGATIVE ACCURACY ##
	# return 1 - get_accuracy(net, x, truth, activation=activation)

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

function take_synaptic_step(net::NeuralNetwork, rewards::Array{Float64, 1}, pol::Policy,
							exploration_probability, ϵ, α; use_bias=false)
	"""
	Function that uses `policy` to
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
					   batch_interval::Int64=30, ϵ = 0.001, α=0.01, γ  = 0.9, activation = "relu")
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
    losses = zeros(N_HIST).+CE_loss(net, x[:,batch_mask], truth[:,batch_mask])

    println("Starting training loop");
	# for i = 1:num_iterations
	@showprogress 1 "Computing..." for i = 1:num_iterations
		if(i % batch_interval == 0)
			batch_mask = shuffle(batch_mask)
		end

        append!(rewards, 0) # appending a zero to the end of the reward array to
                            # record the reward in this iteration.
		take_synaptic_step(net, rewards, pol, exploration_probability, ϵ, α)
        # Calculating reward for this iteration through all synapses.
		# println("CE_loss Time: ")
		# println("Iteration: ",i)
        new_loss = CE_loss(net, x[:, batch_mask], truth[:, batch_mask], activation=activation);

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
    return rewards, losses, net, pol, bias_pol
end

function multiple_forward(X, truth, net, activation::String; num::Int64=10)
	"""
	Test Function for repeatedly timing CE_loss and other functions using MAX FORWARD.
	"""
	for i = 1:num
		@time new_loss = CE_loss(net, X, truth, activation=activation);
	end
end

function run_speed_experiment(verbose::Bool; net_size::Array{Int64,1}=[784,1000,10], iterations::Int64=20000,
							  activation::String="relu")
	BLAS.set_num_threads(6) # This actually makes a measurable difference.
	println("Getting Dataset...");
	@time X_train, Y_train, X_valid, Y_valid = get_X_Y();
	println("Generating Network of size ", net_size,"...");
	@time net = generate_network(net_size);

	println("Testing mass forward on network with X_valid...")
	multiple_forward(X_valid, Y_valid, net, activation, num=10);
	println("\nTesting mass forward on network with X_train...")
	multiple_forward(X_train, Y_train, net, activation, num=10);
	println("\nSize of X_train: ", sizeof(X_train))
	println("Size of X_valid:   ", sizeof(X_valid))
	println("\nSize of X_train: ", size(X_train))
	println("Size of X_valid:   ", size(X_valid))

	println("\n")

end

function run_experiment(verbose::Bool; net_size::Array{Int64,1}=[784,1000,10], iterations::Int64=20000,
						activation::String="relu", explore_prob=0.2, policy_path::String="golden_pol2cy.jld2",
						train_policy::Bool=false, batch_size::Int64=300, batch_interval::Int64=30,
						ϵ=0.001, α=0.01, γ=0.9)
	println("JULIA NUM THREADS: ",Threads.nthreads())
	if(verbose)
		println("Setting number of BLAS threads...")
	end
	BLAS.set_num_threads(6);
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

	@time rewards, losses, net, pol, bias_pol = train_network(x, y, 784, iterations, explore_prob, net_size, train_policy=train_policy,
														pol=policy, batch_size=batch_size, batch_interval=batch_interval,
														ϵ=ϵ, α=α, γ=γ, activation=activation)

	# Huh
	println("ACTIVATION: ",activation)
	train_acc = get_accuracy(net, x, y, activation=activation);
	valid_acc = get_accuracy(net, X_valid, Y_valid, activation=activation);
	train_loss = CE_loss(net, x, y, activation=activation);
	valid_loss = CE_loss(net, X_valid, Y_valid, activation=activation);

	if verbose
	    println("=== RESULTS ===")
	    println("Accuracy on training set: \t", train_acc);
	    println("Accuracy on validation set: \t", valid_acc);
	    println("Final training loss: \t\t", train_loss);
	    println("Final validation loss: \t\t", valid_loss);

	    display(plot(losses))
	end
end

@fastmath run_experiment(true, iterations=5000, batch_size=5000, batch_interval=1000,
						net_size=[784,10,10], activation="relu", α=0.1, ϵ=0.01, explore_prob=0.3,
						);



# train_policy=true, policy_path="None"
#sdf
