# SynRL: Synaptic Reinforcement Learning

Code for the paper _Gradient-Free Neural Network Training via Synaptic-Level Reinforcement Learning_. The paper is currently available on [arXiv](https://arxiv.org/abs/2105.14383).

## Key Files

### Novel Work
 - _Policy file:_ `src/golden_pol2cy.jld2` contains the tabular policy matrix generated in `src/04 Multilayer Perceptron.ipynb`.
 - _Simulated decision boundary experiments:_ `src/04 Multilayer Perceptron.ipynb` contains code for reproducing the neural network results trained on simulated decision boundaries. 
 - _Synaptic Reinforcement Learing Library:_ `src/SynRLv6.jl` is the final set of library functions invoked to train and validate neural networks using the proposed methodology.
 - _OCR Experiment Script:_ `src/OCR_01.jl` is a script that performs an OCR classification experiment using the SynRL library and the hyperparameters passed in through the command line. Results are cached according to command line arguments as well.
 - _Experiment Orchestration Script:_ `src/orch_02.py` is a python script that repeatedly invokes `ORC_01.jl` with different hyperparameters to run a large set of experiments. 

### Baseline
 - _Single Layer Perceptron:_ `src/A1.1 TF Optimized SLP.ipynb` contains code used to train and validate the single-layer perceptron using gradient descent on the notMNIST dataset. 
 - _Multilayer Perceptron:_ `src/A2 MLP Gradient Descent.ipynb` contains code used to train and validate the multi-layer perceptron using gradiennt descent on the notMNNIST dataset. 
