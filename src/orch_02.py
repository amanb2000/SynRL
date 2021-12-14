"""
=== GCE OCR TEST ORCHESTRATION SCRIPT ===

This script is a master orchestration script for testing the `OCR_01.jl` script en masse.

The main goal is to iterate through the many possible hyperparameters for training the neural network using `SynRLv3.jl` and record the results. Records of results are maintained by `OCR_01.jl` as long as an appropriate save location is specified for the JSON record output.

# RELEVANT HYPERPARAMS #

Tier 1: 256 experiments
    - epsilon (NN learning rate): {0.01 -> 0.05 -> 0.1 -> 0.001}
    - policy: {golden_pol2cy -> None}
    - batch_size: {100, 500, 1000, 5000}
    - batch_interval: {250, 700, 1000, 6000}
    - hidden_units: {100, 1000}

Tier 2: [256] * 36 = 6656 experiments if naive
    - (For golden_pol2cy) train_policy: {false -> true}
    - exp_prob: {0.2 -> 0.1 -> 0.4}
    - gamma (future reward discount): {0.9 -> 0.5 -> 1}
        - Only applies when training a new policy.
    - alpha (Q-learning rate): {0.01 -> 0.1}
        - Only applies when training a new policy.

FIXED HYPERPARAMS
    - w_range: default = 1.0
    - n_hist: default = 2 (potentially come back to this one... re-run entire experiment set with a bunch of different values to see how much synapse memory actually affects performance).
    - activation function: Just leave as ReLU, potentially come back to this one... re-run entire experiment with a bunch of different activations.
    - num_iters: default = 100000
    - verbose: SET TO 1, pipe output to a convenient text file saved alongside the JSON.

# SCRIPT DESIGN/SCHEDULING POLICY #

This script will use the subprocess module's `run()` function to invoke scripts.

"""
import subprocess, shlex, time, datetime, json



def main():
    BATCH_NO = "01"
    test = False
    job_limit = 17 # number of allowable simultaneous jobs
    ### Basic `subprocess.run` -- Waits for child process, not parallelized ###
    # result = subprocess.run("ls", capture_output = True)
    # print(result.stdout)

    """ITERATE THROUGH THESE"""
    epsilons = (0.01, 0.05, 0.1, 0.001)
    policies = ("golden_pol2cy.jld2", None)
    # FOR PRELIMINARY TESTS: ITERATE THESE TOGETHER
    batch_sizes = (100, 500, 1000, 5000)
    batch_intervals = (250, 700, 1000, 6000)
    # END TOGETHER ITERATION
    hidden_units = (0, 100, 1000)

    cnt = 0

    log_dict = {"experiment_list": [] }
    jobs = []

    exclusion_list = [10, 13, 16, 19, 1, 22, 4, 7]

    for epsilon in epsilons:
        for policy in policies:
            for i in range(len(batch_sizes)):
                batch_size = batch_sizes[i]
                batch_interval = batch_intervals[i]
                for hidden_unit in hidden_units:
                    cnt+=1

                    # Exclusion list maintenance -- cannot repeat experiments from previously!
                    if cnt in exclusion_list:
                        break

                    command_string = "julia OCR_01.jl "
                    command_string += "--savename ../experiments/batch{}/exp{}.json ".format(BATCH_NO, cnt)
                    command_string += "--epsilon {} ".format(epsilon)
                    if policy != None:
                        command_string += "--policy {} ".format(policy)
                    command_string += "--batch_size {} ".format(batch_size)
                    command_string += "--batch_interval {} ".format(batch_interval)
                    command_string += "--hidden_units {} ".format(hidden_unit)
                    
                    ###########################
                    ########### TEST ##########
                    ###########################
                    if test:
                        command_string += "--num_iters 4 "
                    else:
                        command_string += "--num_iters 100000 "
                
                    print("Starting experiment {}/96".format(cnt))
                    # print("\t",command_string)
               
                    experiment_dict = {
                        "epsilon": epsilon,
                        "policy": policy,
                        "batch_size": batch_size,
                        "batch_interval": batch_interval,
                        "hidden_units": hidden_unit,
                        "name": "experiments/batch{}/exp{}.json ".format(BATCH_NO, cnt)
                    }
                    log_dict["experiment_list"].append(experiment_dict)
                    # print(experiment_dict)

                    args = shlex.split(command_string)

                    """RUNNING THE SCRIPT!"""
                    # p = subprocess.Popen(args)
                    p = subprocess.Popen(args, stdout=subprocess.DEVNULL)
                    jobs.append(p)
                    if test:
                        break


                    while True:
                        time.sleep(10)
                        sub_cnt = 0
                        for p in jobs:
                            if p.poll() == None:
                                sub_cnt += 1
                        print('\nStarted {}/{} Jobs @ {}'.format(cnt, 96, datetime.datetime.now()))
                        print("\tCurrently running {}/{} Allowable Jobs".format(sub_cnt, job_limit))
                        if sub_cnt < job_limit:
                            break
 
                if test:
                    break
            if test:
                break
        if test:
            break

    print("Writing Experiment Log File...")
    with open('../experiments/batch{}.txt'.format(BATCH_NO), 'w') as file:
        file.write(json.dumps(log_dict)) # use `json.loads` to do the reverse

    finished = False

    while not finished:
        time.sleep(10)
        cnt = 0
        done = 0
        for p in jobs:
            cnt += 1
            if p.poll() != None:
                done += 1
        print('Finished {}/{} Jobs @ {}'.format(done, cnt, datetime.datetime.now()))
        if cnt == done:
            break
    print("============================")
    print("=== DONE WITH EXPERIMENT ===")
    print("============================")


if __name__ == "__main__":
    main()
