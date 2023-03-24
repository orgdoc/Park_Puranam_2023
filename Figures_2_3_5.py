# Vicarious Learning without Knowledge Differentials Ver 1.0

# Importing modules
import math
import random
import numpy as np
import csv

#####################################################################################################
# SET SIMULATION PARAMETERS HERE
T = 1000                 # number of periods to simulate the model
sampleSize = 10000       # sample size
M = 50                   # number of alternatives
Figure = 5               # figure to replicate
                         # 2 for Figure 2
                         # 3.1 for Figure 3 (tau = 0); 3.2 for Figure 3 (tau = 0.01); 3.3 for Figure 3 (tau = 0.03)
                         # 5 for Figure 5

# Enter inputs
if Figure == 2:
    choiceRule = 0              # if 0, then agents follow soft-max rule with tau -> 0.
                                # if 1, then agents follow soft-max rule with tau > 0.
    ownActionDependence = 1     # if 0, feedback for all actions is available.
                                # if 1, feedback for unchosen actions is NOT available.
elif Figure == 3.1:
    choiceRule = 0              # if 0, then agents follow soft-max rule with tau -> 0.
                                # if 1, then agents follow soft-max rule with tau > 0.
    ownActionDependence = 1     # if 0, feedback for all actions is available.
                                # if 1, feedback for unchosen actions is NOT available.
elif Figure == 3.2:
    choiceRule = 1              # if 0, then agents follow soft-max rule with tau -> 0.
                                # if 1, then agents follow soft-max rule with tau > 0.
    tau = 0.01                  # Soft-max temperature for tau > 0
    ownActionDependence = 1     # if 0, feedback for all actions is available.
                                # if 1, feedback for unchosen actions is NOT available.
elif Figure == 3.3:
    choiceRule = 1              # if 0, then agents follow soft-max rule with tau -> 0.
                                # if 1, then agents follow soft-max rule with tau > 0.
    tau = 0.03                  # Soft-max temperature for tau > 0
    ownActionDependence = 1     # if 0, feedback for all actions is available.
                                # if 1, feedback for unchosen actions is NOT available.
elif Figure == 5:
    choiceRule = 0              # if 0, then agents follow soft-max rule with tau -> 0.
                                # if 1, then agents follow soft-max rule with tau > 0.
    ownActionDependence = 0     # if 0, feedback for all actions is available.
                                # if 1, feedback for unchosen actions is NOT available.

######################################################################################################
# DEFINING FUNCTIONS
def genEnvironment(M):  # Generate task environment
    r = np.random.rand(M)
    return r
def genPriors(M): # Generate random priors
    r = np.random.rand(M)
    return r
def getBest(reality):  # find the best action
    best = reality.argmax(axis=0)
    return best
def hardmax(attraction, M):  # max action selection
    choice = attraction.argmax(axis=0)
    return choice
def softmax(attraction, M, tau):  # softmax action selection
    prob = np.zeros((1, M))
    denom = 0
    i = 0
    while i < M:
        denom = denom + math.exp((attraction[i]) / tau)
        i = i + 1
    roulette = random.random()
    i = 0
    p = 0
    while i < M:
        prob[0][i] = math.exp(attraction[i] / tau) / denom
        p = p + prob[0][i]
        if p > roulette:
            choice = i
            return choice
            break  # stops computing probability of action selection as soon as cumulative probability exceeds roulette
        i = i + 1

######################################################################################################
# DEFINING OBJECTS
# Task environment
reality = np.zeros(M)

# Agents' beliefs (no vicarious learning = 1 and 2; observational learning = 3 and 4; belief sharing = 5 and 6)
attraction1 = np.zeros(M)
attraction2 = np.zeros(M)
attraction3 = np.zeros(M)
attraction4 = np.zeros(M)
attraction5 = np.zeros(M)
attraction6 = np.zeros(M)
attractionlag5 = np.zeros(M)
attractionlag6 = np.zeros(M)

# To keep track of count of # times action selected
count1 = np.zeros(M)
count2 = np.zeros(M)
count3 = np.zeros(M)
count4 = np.zeros(M)
count5 = np.zeros(M)
count6 = np.zeros(M)

# Defining results vectors
avg_agent_perf1 = np.zeros((T, sampleSize))
avg_agent_perf2 = np.zeros((T, sampleSize))
avg_agent_perf3 = np.zeros((T, sampleSize))
both_correct1 = np.zeros((T, sampleSize))
both_correct2 = np.zeros((T, sampleSize))
both_correct3 = np.zeros((T, sampleSize))
convergence1 = np.zeros((T, sampleSize))
convergence2 = np.zeros((T, sampleSize))
convergence3 = np.zeros((T, sampleSize))


######################################################################################################
# SIMULTAION IS RUN HERE
for a in range(sampleSize):
    # Initialize beliefs and task environments
    reality = genEnvironment(M)
    attraction1 = genPriors(M)
    attraction2 = genPriors(M)
    attraction3 = genPriors(M)
    attraction4 = genPriors(M)
    attraction5 = genPriors(M)
    attraction6 = genPriors(M)
    count1 = np.ones(M)
    count2 = np.ones(M)
    count3 = np.ones(M)
    count4 = np.ones(M)
    count5 = np.ones(M)
    count6 = np.ones(M)
    bestchoice = getBest(reality)

    for t in range(T):
        # Action selection
        if choiceRule == 0:
            choice1 = hardmax(attraction1, M)
            choice2 = hardmax(attraction2, M)
            choice3 = hardmax(attraction3, M)
            choice4 = hardmax(attraction4, M)
            choice5 = hardmax(attraction5, M)
            choice6 = hardmax(attraction6, M)
        elif choiceRule == 1:
            choice1 = softmax(attraction1, M, tau)
            choice2 = softmax(attraction2, M, tau)
            choice3 = softmax(attraction3, M, tau)
            choice4 = softmax(attraction4, M, tau)
            choice5 = softmax(attraction5, M, tau)
            choice6 = softmax(attraction6, M, tau)

        # Performance feedback
        payoff1 = reality[choice1]
        payoff2 = reality[choice2]
        payoff3 = reality[choice3]
        payoff4 = reality[choice4]
        payoff5 = reality[choice5]
        payoff6 = reality[choice6]

        # Record average performance
        avg_agent_perf1[t][a] = (payoff1 + payoff2) / 2
        avg_agent_perf2[t][a] = (payoff3 + payoff4) / 2
        avg_agent_perf3[t][a] = (payoff5 + payoff6) / 2

        # Record joint search success
        if (choice1 == bestchoice) & (choice2 == bestchoice):
            both_correct1[t][a] = 1
        if (choice3 == bestchoice) & (choice4 == bestchoice):
            both_correct2[t][a] = 1
        if (choice5 == bestchoice) & (choice6 == bestchoice):
            both_correct3[t][a] = 1

        # Record convergence in actions
        if choice1 == choice2:
            convergence1[t][a] = 1
        if choice3 == choice4:
            convergence2[t][a] = 1
        if choice5 == choice6:
            convergence3[t][a] = 1

        # Update belief under own-action dependence
        if ownActionDependence == 1:
            # Agents without vicarious learning
            attraction1[choice1] = count1[choice1] * attraction1[choice1] / (count1[choice1] + 1) + payoff1 / (count1[choice1] + 1)
            attraction2[choice2] = count2[choice2] * attraction2[choice2] / (count2[choice2] + 1) + payoff2 / (count2[choice2] + 1)
            count1[choice1] += 1
            count2[choice2] += 1

            # Agents with observational learning
            attraction3[choice3] = count3[choice3] * attraction3[choice3] / (count3[choice3] + 1) + payoff3 / (count3[choice3] + 1)
            attraction4[choice4] = count4[choice4] * attraction4[choice4] / (count4[choice4] + 1) + payoff4 / (count4[choice4] + 1)
            count3[choice3] += 1
            count4[choice4] += 1
            attraction3[choice4] = count3[choice4] * attraction3[choice4] / (count3[choice4] + 1) + payoff4 / (count3[choice4] + 1)
            attraction4[choice3] = count4[choice3] * attraction4[choice3] / (count4[choice3] + 1) + payoff3 / (count4[choice3] + 1)
            count3[choice4] += 1
            count4[choice3] += 1

            # Agents with belief sharing
            attraction5[choice5] = count5[choice5] * attraction5[choice5] / (count5[choice5] + 1) + payoff5 / (count5[choice5] + 1)
            attraction6[choice6] = count6[choice6] * attraction6[choice6] / (count6[choice6] + 1) + payoff6 / (count6[choice6] + 1)
            count5[choice5] += 1
            count6[choice6] += 1
            attractionlag5 = np.copy(attraction5)
            attractionlag6 = np.copy(attraction6)
            attraction5 = 0.5 * attractionlag5 + 0.5 * attractionlag6
            attraction6 = 0.5 * attractionlag6 + 0.5 * attractionlag5

        # Update belief when feedback of unchosen actions is available
        elif ownActionDependence == 0:
            # Update beliefs for all actions
            for i in range(M):
                attraction1[i] = count1[i] * attraction1[i] / (count1[i] + 1) + reality[i] / (count1[i] + 1)
                attraction2[i] = count2[i] * attraction2[i] / (count2[i] + 1) + reality[i] / (count2[i] + 1)
                attraction3[i] = count3[i] * attraction3[i] / (count3[i] + 1) + reality[i] / (count3[i] + 1)
                attraction4[i] = count4[i] * attraction4[i] / (count4[i] + 1) + reality[i] / (count4[i] + 1)
                attraction5[i] = count5[i] * attraction5[i] / (count5[i] + 1) + reality[i] / (count5[i] + 1)
                attraction6[i] = count6[i] * attraction6[i] / (count6[i] + 1) + reality[i] / (count6[i] + 1)
                count1[i] += 1
                count2[i] += 1
                count3[i] += 1
                count4[i] += 1
                count5[i] += 1
                count6[i] += 1

            # Agents with observational learning
            attraction3[choice4] = count3[choice4] * attraction3[choice4] / (count3[choice4] + 1) + payoff4 / (count3[choice4] + 1)
            attraction4[choice3] = count4[choice3] * attraction4[choice3] / (count4[choice3] + 1) + payoff3 / (count4[choice3] + 1)
            count3[choice4] += 1
            count4[choice3] += 1

            # Agents with belief sharing
            attractionlag5 = np.copy(attraction5)
            attractionlag6 = np.copy(attraction6)
            attraction5 = 0.5 * attractionlag5 + 0.5 * attractionlag6
            attraction6 = 0.5 * attractionlag6 + 0.5 * attractionlag5

result_org = np.zeros((T, 10))

for t in range(T):  # Compiling final output
    result_org[t, 0] = t + 1
    result_org[t, 1] = float(np.sum(avg_agent_perf1[t, :])) / sampleSize
    result_org[t, 2] = float(np.sum(avg_agent_perf2[t, :])) / sampleSize
    result_org[t, 3] = float(np.sum(avg_agent_perf3[t, :])) / sampleSize
    result_org[t, 4] = float(np.sum(both_correct1[t, :])) / sampleSize
    result_org[t, 5] = float(np.sum(both_correct2[t, :])) / sampleSize
    result_org[t, 6] = float(np.sum(both_correct3[t, :])) / sampleSize
    result_org[t, 7] = float(np.sum(convergence1[t, :])) / sampleSize
    result_org[t, 8] = float(np.sum(convergence2[t, :])) / sampleSize
    result_org[t, 9] = float(np.sum(convergence3[t, :])) / sampleSize

######################################################################################################
# WRITING RESULTS TO CSV FILE
if Figure == 2:
    filename = ("Figure2.csv")
    with open(filename, 'w', newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(
            ['Period', 'Performance (without VL)', 'Performance (OL)', 'Performance (BS)', 'Search Success (without VL)', 'Search Success (OL)', 'Search Success (BS)', 'Convergence (without VL)', 'Convergence (OL)', 'Convergence (BS)'])
        for values in result_org:
            thewriter.writerow(values)
        f.close()
elif Figure == 3.1:
    filename = ("Figure3a.csv")
    with open(filename, 'w', newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(
            ['Period', 'Performance (without VL)', 'Performance (OL)', 'Performance (BS)',
             'Search Success (without VL)', 'Search Success (OL)', 'Search Success (BS)', 'Convergence (without VL)',
             'Convergence (OL)', 'Convergence (BS)'])
        for values in result_org:
            thewriter.writerow(values)
        f.close()
elif Figure == 3.2:
    filename = ("Figure3b.csv")
    with open(filename, 'w', newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(
            ['Period', 'Performance (without VL)', 'Performance (OL)', 'Performance (BS)', 'Search Success (without VL)', 'Search Success (OL)', 'Search Success (BS)', 'Convergence (without VL)', 'Convergence (OL)', 'Convergence (BS)'])
        for values in result_org:
            thewriter.writerow(values)
        f.close()
elif Figure == 3.3:
    filename = ("Figure3c.csv")
    with open(filename, 'w', newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(
            ['Period', 'Performance (without VL)', 'Performance (OL)', 'Performance (BS)', 'Search Success (without VL)', 'Search Success (OL)', 'Search Success (BS)', 'Convergence (without VL)', 'Convergence (OL)', 'Convergence (BS)'])
        for values in result_org:
            thewriter.writerow(values)
        f.close()
elif Figure == 5:
    filename = ("Figure5.csv")
    with open(filename, 'w', newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(
            ['Period', 'Performance (without VL)', 'Performance (OL)', 'Performance (BS)', 'Search Success (without VL)', 'Search Success (OL)', 'Search Success (BS)', 'Convergence (without VL)', 'Convergence (OL)', 'Convergence (BS)'])
        for values in result_org:
            thewriter.writerow(values)
        f.close()

