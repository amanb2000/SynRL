"""
Goal: create a callable script to run experiments en-masse on a Google compute cluster.

Modifyable parameters: FROM BEFORE
 - W_RANGE: Range of neural network non-bias weights upon instantiation.
    - default: 1
 - N_HIST: Number of history terms in the policy.
    - default: 2
 - ϵ: NN weight update rate (
    - default: 0.001
 - α: Q-learning rate
    - default 0.01 or 0.001
 - γ: Future reward discount for Q-learning
    - default: 0.9

Modifyable parameters: NEW
 - Number of hidden units.
    - default: 100?
 - Minibatch size.
    - default: -1 (indicating full batch...)
    - Should use `randperm!(ones_zeros_length_N)` to pick them...
 - Given policy.
    - default: -1 (generate your own new policy and train it)
    - Otherwise it should be the directory in which it's been cached.
 - Train policy.
    - default: `true`.
 - Output directory + output file name (see next section)

Output format:
 Given a directory and a file name to write to, the output should include:
 - Final CE loss on train/test/validation set.
 - Final accuracy on train/test/validation set.
 - Cached loss over time (saved with UID generated name, name is written in record).
 - Time taken to complete training.
 - Cached version of the final policy IF the policy was trained/edited/created.
 - All given hyperparams.
"""

println("Loading imports...")
# IMPORT BOX #
using LinearAlgebra
using Random
using StatsFuns
using Statistics
using ProgressMeter
using NPZ
using Plots
using FileIO, JLD2

import JSON

using Profile

using ArgParse
println("Done loading imports")
#######################################
### Getting command line arguments. ###
#######################################
s = ArgParseSettings()
@add_arg_table s begin
    "--activation"
        help = "Name of the activation function. Currently supports `relu` and `tanh`. Defaults to `relu`."
        arg_type = String
        default = "relu"
    "--savename"
        help = "path/name of json file to save results in (e.g., experiments/exp001.json)."
        arg_type = String
        default = "../experiments/exp001.json"
    "--batch_size"
        help = "size of each mini-batch during training (number of samples)"
        arg_type = Int64
        default = 1000
    "--batch_interval"
        help = "number of iterations for which a given batch is used before refresh"
        arg_type = Int64
        default = 500
    "--w_range"
        help = "range (+/-) for NN non-bias weight instantiation (uniform distr.)"
        arg_type = Float64
        default = 1.0
    "--n_hist"
        help = "number of history terms in policy for past rewards and actions."
        arg_type = Int
        default = 2
    "--epsilon"
        help = "neural network learning rate using synaptic policy."
        arg_type = Float64
        default = 0.01
    "--alpha"
        help = "q-learning learning rate."
        arg_type = Float64
        default = 0.01
    "--gamma"
        help = "future reward discount."
        arg_type = Float64
        default = 0.9
    "--hidden_units"
        help = "Number of hidden units in neural network structure. -1 for none."
        arg_type = Int
        default = 100
    "--policy_path"
        help = "Path to an existing policy, if any. Defaults to `None` (i.e., train a new policy)."
        arg_type = String
        default = "None"
    "--train_policy"
        help = "Whether or not policy should be updated. Defaults to 1 (i.e., do update)."
        arg_type = Int
        default = 1
    "--num_iters"
        help = "Number of iterations to run training."
        arg_type = Int
        # default = 100000
        default = 100
    "--exp_prob"
        help = "Probability of a random move for each synapse at each iteration."
        arg_type = Float64
        default = 0.2
    "--verbose"
        help = "1 for verbose, default to 0."
        arg_type = Int
        default = 0
end

parsed_args = parse_args(ARGS, s)

###################################
### Assigning parsed arguments. ###
###################################

const W_RANGE = parsed_args["w_range"] # Range for neural network non-bias weight instantiation (+/- W_RANGE).
const N_HIST = parsed_args["n_hist"] # Number of `action` and `rewards` in the history buffers. 2 has worked surprisingly well.
const ϵ = parsed_args["epsilon"] # Weight update rate
const α = parsed_args["alpha"] # Q-learning rate
const γ = parsed_args["gamma"] # Future reward discount factor
const iters = parsed_args["num_iters"]
const hidden_units = parsed_args["hidden_units"]
const policy_path = parsed_args["policy_path"]
const train_policy = (parsed_args["train_policy"] == 1)
const exp_prob = parsed_args["exp_prob"]

const savename = parsed_args["savename"]
const batch_size = parsed_args["batch_size"]
const batch_interval = parsed_args["batch_interval"]

const activation = parsed_args["activation"]

verbose = false
if parsed_args["verbose"] != 0
    verbose = true
end

###################################
### Importing Library Functions ###
###################################

if verbose
    println("Loading SynRLv3...")
end
include("SynRLv3.jl")

if verbose
    println("Done loading SynRLv3.\n")
end

#######################
### Creating Policy ###
#######################

if verbose
    println("Getting policy...")
end
if policy_path != "None"
    policy = FileIO.load(policy_path, "pol");
    if verbose
        println("Policy loaded\n")
    end
else
    policy = new_policy();
    if verbose
        println("New policy generated\n")
    end
end


#######################
### Getting Dataset ###
#######################

include("notMNIST_util.jl") # Functions: get_X_Y, get_accuracy(network, x, y)
if verbose
    println("Loading training + validation data...")
end
X_train_, Y_train_, X_valid_, Y_valid_ = get_X_Y();

const X_train = X_train_;
const Y_train = Y_train_;
const X_valid = X_valid_;
const Y_valid = Y_valid_;


if verbose
    println("Done loading data.")
    println("X_train size: ", size(X_train))
    println("Y_train size: ", size(Y_train))
end

const dim = 784
const iterations = iters
const explore = exp_prob
if(hidden_units > 0)
    net_shape = [dim, hidden_units, 10]
else
    net_shape = [dim, 10]
end

if verbose
    println("Beginning experiment...")
end

rewards, losses, net, pol, bias_pol = Juno.@profiler train_network(X_train, Y_train, dim, iterations,
                                                    explore, net_shape, pol=policy, train_policy=train_policy,
                                                    batch_size=batch_size, batch_interval=batch_interval);

# Calculating Final Accuracies

train_acc = get_accuracy(net, X_train, Y_train);
valid_acc = get_accuracy(net, X_valid, Y_valid);
train_loss = CE_loss(net, X_train, Y_train);
valid_loss = CE_loss(net, X_valid, Y_valid);

if verbose
    println("=== RESULTS ===")
    println("Accuracy on training set: \t", train_acc);
    println("Accuracy on validation set: \t", valid_acc);
    println("Final training loss: \t\t", train_loss);
    println("Final validation loss: \t\t", valid_loss);

    plot(losses)

end
# TODO: Write code for final outputting.

# output_dict generation

output_dict = Dict{String, Any}(
    "W_RANGE" => W_RANGE,
    "N_HIST" => N_HIST,
    "epsilon" => ϵ,
    "alpha" => α,
    "gamma" => γ,
    "iters" => iters,
    "hidden_units" => hidden_units,
    "policy_path" => policy_path,
    "train_policy" => train_policy,
    "exp_prob" => exp_prob,
    "savename" => savename,
    "batch_size" => batch_size,
    "batch_interval" => batch_interval,
    "losses" => losses,
    "train_loss" => train_loss,
    "valid_loss" => valid_loss,
    "train_acc" => train_acc,
    "valid_acc" => valid_acc,
);

output_string = JSON.json(output_dict)

open(savename, "w") do f
    write(f, output_string)
end
