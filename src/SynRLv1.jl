"""
===================
=== SYNRL mk. 1 ===
===================

This is the first version of the synaptic reinforcement learning library.

---
Author: Aman Bhargava
		University of Toronto
		Engineering Science
		Class of 2022
Date:   March 21, 2021
---

This implements a traditional multi-layer perceptron system trained using a (seemingly novel) synaptic reinforcement learning method. The goal is to create an easy-to-use fast library for running training tests.

DEPENDENCIES:
```
using LinearAlgebra
using Plots
using Plotly
using Random
using StatsFuns
using Statistics

using ProgressMeter
```


OPTIONAL: 
 - ProgressMeter

USE INSTRUCTIONS:
 - Make sure that the following constants are specified in the script before calling 
   "include("SynRLv1.jl"):
```
DIM = 10 # Dimensionality of augmented data vectors (data has d-1 dimensions)
W_RANGE = 1 # Range for neural network non-bias weight instantiation (+/- W_RANGE).
N_HIST = 2 # Number of `action` and `rewards` in the history buffers. 2 has worked surprisingly well.
ϵ = 0.001 # Weight update rate
α = 0.01 # Q-learning rate
γ = 0.9 # Future reward discount factor
# --- #
X_RANGE = 10 # Range for values in training data (~uniform(+/-)) -- used for data generation.
N = 200 # Number of datapoints in a given batch of generated data -- used for data generation.
```
 - Would be advisable to include these as command line arguments.
"""




"""
=== Loading Structs ===
 - Neuron
 - Layer
 - NeuralNetwork
 - Policy
"""

mutable struct Neuron
    w::Array{Float64, 1}
    actions::Array{Int8, 2}
end

mutable struct Layer
    neurons::Array{Neuron, 1}
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
    Q::Array{Float64, N_HIST*2+1} # for each history round we have `actions` and `rewards`
end


"""
=== Forward Functions === 
"""

function forward(x, neuron::Neuron)
    return tanh(transpose(x)*neuron.w)
end

function forward(x, layer::Layer)
    out = zeros(Float64, length(layer.neurons))
    x = [1; x]; # Adding a bias term
    for i = 1:length(layer.neurons)
        out[i] = forward(x, layer.neurons[i])
    end
    return out
end

function forward(x, network::NeuralNetwork)
    intermediate = x;
    for i = 1:length(network.layers)
        intermediate = forward(intermediate, network.layers[i]);
    end
    return intermediate
end

function mass_forward(x, network::NeuralNetwork)
    """ 
    Function for processing a batch of data through a neural network.
    x should be a matrix of column vectors where each column is a single piece of data. 
    y will be of the same format with column vectors representing the output.
    """
    y = zeros(network.shape[end], size(x)[2])
    
    for i = 1:size(x)[2]
        y[:,i] = forward(x[:,i], network)
    end
    
    return y
end




"""
=== Generate Functions ===
"""

function generate_neuron(dim::Int64, zero_bias::Bool=true)
    # Generate history matrix
    actions = zeros(dim, N_HIST+1).+2 # For DIM synapses we store their actions.
                                    # Starting with N_HIST actions set to 0.
    actions = convert(Array{Int8, 2},actions)
    
    # Generate weight vector
    w = rand(Float64, dim);
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


function generate_layer(num_neurons, dim_neuron; zero_bias::Bool=true)
    """
    num_neurons: Number of neurons in the layer
    dim_neuron:  Number of neurons in the previous layer (including bias).
    """
    neuron_array = Array{Neuron,1}(undef, num_neurons);
    for i = 1:num_neurons
        neuron_array[i] = generate_neuron(dim_neuron, zero_bias);
    end
    return Layer(neuron_array);
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


function generate_data(DIM, x_rng, N)
    x = rand(Float64, (DIM, N));
    x *= 2;
    x = x .- 1;
    x *= x_rng;
#     x[1,:] .= 1; # Bias term
    return x;
end



"""
Visualization Functions
 - TODO: Set up a plot saving system.
"""

function show_2d_boundary(net::NeuralNetwork; xyrange = 2, num_points = 100, title="Decision Boundary R2 → (1,0)")
    """
    Shows output/decision boundary of neural network. Must map R^2 => [-1,1].
    net:        NeuralNetwork struct in question.
    xyrange:    Coordinate range for neural network (domain of input +/- range).
    num_points: Number of total data points (uniformly distributed in region).
    """
    
    xy = generate_data(2, xyrange, num_points)
    
    z = zeros(num_points)
    
    for i = 1:num_points
        z[i] = forward(xy[:,i], net)[1];
    end
    
    Plots.scatter( (xy[1,:])[z .> 0], (xy[2,:])[z .> 0], label = "+")
    Plots.scatter!( (xy[1,:])[z .< 0], (xy[2,:])[z .< 0], label = "-")
    Plots.title!(title)
end



"""
=== Loss and Training Functions ===
"""

function loss_linear(net, x, truth)
    pred1 = mass_forward(x, net);
    diff = pred1-truth;
    return norm(diff)/length(pred1);
end


function train_network(x, truth, dim::Int64, iters::Int64, explore_prob::Float64, 
                       net_shape::Array{Int64, 1}; train_policy::Bool=true, pol::Any=false, 
                       bias_pol::Any=false, use_bias::Bool=false)
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
    
    Outputs
    ---
    actions, rewards, losses, percep, pol, bias_pol
    """
    
    num_iterations = iters;
    exploration_probability = explore_prob;
    net = generate_network(net_shape); 
    
    if pol == false || typeof(pol) != Policy
        pol = new_policy();
        bias_pol = new_policy();
        train_policy = true;
    end
    
    rewards = zeros(N_HIST).+2 # Rewards history is instantiated with N_HIST zeros.
    losses = zeros(N_HIST).+loss_linear(net, x, truth)
 
    println("Starting training loop");
    @showprogress 1 "Computing..." for i = 1:num_iterations
        append!(rewards, 0) # appending a zero to the end of the reward array to 
                            # record the reward in this iteration.
        for layer in net.layers 
            for neuron in layer.neurons
                
                neuron.actions = circshift(neuron.actions, (0, -1)) # Shifting each row of the neuron's history 
                                                                   # one to the left
                neuron.actions[:,end] .= 0 # Setting the element that looped around to the end to zero
                                           # (this is the element that will be populated here)
                # Iterating through each synapse and taking an action (inc/dec/same).
                for j = 1:length(neuron.w)
                    # The address for synapse j consists of the last N_HIST actions concatenated
                    # with the last N_HIST rewards.
                    # Example: For N_HIST = 2, addr = [action_{t-1}, action_{t-2}, reward_{t-1}, reward_t-2]
                    # Since actions and rewards take on values of either 1, 2, or 3 (indicating 
                    # decrease, stay the same, or increase), one can use them to index an array in 
                    # Julia (a 1-indexed language).
#                     addr = [actions[j, end-N_HIST:end-1]; rewards[end-N_HIST:end-1]]
                    addr = [neuron.actions[j, end-N_HIST:end-1]; rewards[end-N_HIST:end-1]] # TODO: Test this.
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
                    neuron.actions[j,end] = a;

                    # Taking the action by adjusting the synapse weight accordingly.
                    if j == 1 
                        neuron.w[j] += (a-2)*ϵ*2; # TEST: Using smaller learning rate for bias term
                    else
                        neuron.w[j] += (a-2)*ϵ;
                    end
                end # iterating through each neuron weight vector entry
            end # Iterating through each neuron in the layer
        end
        # Calculating reward for this iteration through all synapses.
        new_loss = loss_linear(net, x, truth);
        
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
        else
#             println("NOT UPDATING POLICY!!");
        end
    end
    
    # Returning all actions taken, all rewards, all losses
    # And the final perceptron and the final perceptron learning policy.
    return rewards, losses, net, pol, bias_pol
end




# LEGACY FUNCTION #
function match_network(dim::Int64, target::NeuralNetwork, iters::Int64, explore_prob::Float64, 
                       net_shape::Array{Int64, 1}; train_policy::Bool=true, pol::Any=false, 
                       bias_pol::Any=false, use_bias::Bool=false)
    """
    Function to train a new neural network to match performance of a target neural network.
    
    Inputs
    ---
    `dim`             Dimension of the input data.
    `target`          Neural network to match performance with.
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
    
    Outputs
    ---
    actions, rewards, losses, percep, pol, bias_pol
    """
    
    num_iterations = iters;
    exploration_probability = explore_prob;
    net = generate_network(net_shape);
    x = generate_data(dim, X_RANGE, N);
    
    truth = mass_forward(x, target);
    
    if pol == false || typeof(pol) != Policy
        pol = new_policy();
        bias_pol = new_policy();
        train_policy = true;
    end
    
    rewards = zeros(N_HIST).+2 # Rewards history is instantiated with N_HIST zeros.
    losses = zeros(N_HIST).+loss_linear(net, x, truth)
 
    println("Starting training loop");
    @showprogress 1 "Computing..." for i = 1:num_iterations
        append!(rewards, 0) # appending a zero to the end of the reward array to 
                            # record the reward in this iteration.
        for layer in net.layers 
            for neuron in layer.neurons
                
                neuron.actions = circshift(neuron.actions, (0, -1)) # Shifting each row of the neuron's history 
                                                                   # one to the left
                neuron.actions[:,end] .= 0 # Setting the element that looped around to the end to zero
                                           # (this is the element that will be populated here)
                # Iterating through each synapse and taking an action (inc/dec/same).
                for j = 1:length(neuron.w)
                    # The address for synapse j consists of the last N_HIST actions concatenated
                    # with the last N_HIST rewards.
                    # Example: For N_HIST = 2, addr = [action_{t-1}, action_{t-2}, reward_{t-1}, reward_t-2]
                    # Since actions and rewards take on values of either 1, 2, or 3 (indicating 
                    # decrease, stay the same, or increase), one can use them to index an array in 
                    # Julia (a 1-indexed language).
#                     addr = [actions[j, end-N_HIST:end-1]; rewards[end-N_HIST:end-1]]
                    addr = [neuron.actions[j, end-N_HIST:end-1]; rewards[end-N_HIST:end-1]] # TODO: Test this.
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
                    neuron.actions[j,end] = a;

                    # Taking the action by adjusting the synapse weight accordingly.
                    if j == 1 
                        neuron.w[j] += (a-2)*ϵ*2; # TEST: Using smaller learning rate for bias term
                    else
                        neuron.w[j] += (a-2)*ϵ;
                    end
                end # iterating through each neuron weight vector entry
            end # Iterating through each neuron in the layer
        end
        # Calculating reward for this iteration through all synapses.
        new_loss = loss_linear(net, x, truth);
        
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
        else
#             println("NOT UPDATING POLICY!!");
        end
    end
    
    # Returning all actions taken, all rewards, all losses
    # And the final perceptron and the final perceptron learning policy.
    return rewards, losses, net, pol, bias_pol
end

































