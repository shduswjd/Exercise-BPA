import numpy as np
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianNetwork, MarkovNetwork

model = BayesianNetwork()
edges = [
    ('E', 'SE'),
    ('E', 'SH'),
    ('P', 'SE'),
    ('P', 'SH')
]

model.add_edges_from(edges)

cpd = TabularCPD('grade',3,[[0.1,0.1,0.1,0.1,0.1,0.1],
                            [0.1,0.1,0.1,0.1,0.1,0.1],
                            [0.8,0.8,0.8,0.8,0.8,0.8]],
                            evidence=['diff', 'intel'], evidence_card=[2,3])

# Define probabilities E (for Election) and P (for Party) 
# You can use `TabularCPD` function for this

# 사전확률
# E = True / False
# P = party A/ party B/ party C
E = TabularCPD(variable='E', variable_card=2, values=[[0.5], [0.5]])
P = TabularCPD(variable='P', variable_card=3, values=[[1/3], [1/3], [1/3]])

# 조건부 확률
# SE: low, moderate, high
# SH: low, moderate, high
SE = TabularCPD(variable='SE', variable_card=3, values=[
    # low
    [1/3, 0.15, 0.2, 0.89, 0.15, 0.2],
    [1/3, 0.15, 0, 0.1, 0.15, 0],
    [1/3, 0.7, 0.8, 0.01, 0.7, 0.8]
],
evidence=['E', 'P'],
evidence_card=[2, 3])

print(SE)

SH = TabularCPD(variable='SH', variable_card=3, values=[
    [0.05, 1/3, 0.2, 0.05, 0.89, 0.2],
    [0.15, 1/3, 0, 0.15, 0.1, 0],
    [0.8, 1/3, 0.8, 0.8, 0.01, 0.8],
], 
evidence=['E','P'], evidence_card=[2, 3])
print(SH)

model.add_cpds(E, P, SH, SE)

# Checking if the cpds are valid for the model.
model.check_model()

from pgmpy.inference import VariableElimination
solver = VariableElimination(model)
result = solver.query(variables=['P'], evidence={'SE':2, 'SH':1})
print(result)


# Given election is imminent and Party C is not fielding any candidates, 
# you see a candidate is promising a high amount of spending on education and a moderate amount of spending on health care, 
# what is the probability of candidate being in party B?
model = BayesianNetwork()
edges = [
    ('E', 'SE'),
    ('E', 'SH'),
    ('P', 'SE'),
    ('P', 'SH')
]

model.add_edges_from(edges)

E = TabularCPD(variable='E', variable_card=2, values=[[0.5], [0.5]])
P = TabularCPD(variable='P', variable_card=2, values=[[0.5], [0.5]])

SE = TabularCPD(variable='SE', variable_card=3, values=[
    # low
    [1/3, 0.15, 0.89, 0.15],
    [1/3, 0.15, 0.1, 0.15],
    [1/3, 0.7, 0.01, 0.7]
],
evidence=['E', 'P'],
evidence_card=[2, 2])

SH = TabularCPD(variable='SH', variable_card=3, values=[
    [0.05, 1/3, 0.05, 0.89],
    [0.15, 1/3, 0.15, 0.1],
    [0.8, 1/3, 0.8, 0.01],
], 
evidence=['E','P'], evidence_card=[2, 2])

model.add_cpds(E, P, SH, SE)

# Checking if the cpds are valid for the model.
model.check_model()

belief_propagation = BeliefPropagation(model)
res = belief_propagation.query(variables=["P"], evidence = {"SE":2, "SH":1, "E":1})
print(res)

# Burglar Alarm
model = BayesianNetwork()
"""
Build the model adding nodes.
You can use `add_edges_from` function for this
"""
edges = [
('B', 'A'),
('E', 'A'),
('A', 'J'),
('A', 'M')]

# Define the prior probabilities for Burglary and Earthquake.
model.add_edges_from(edges)
B = TabularCPD(variable = "B", variable_card = 2, values = [[0.999], [0.001]])
E = TabularCPD(variable = "E", variable_card = 2, values = [[0.998], [0.002]])

# Define the conditional probabilities for Alarm, John calls, and Mary calls.
A = TabularCPD(variable = "A", variable_card = 2, values = [
    [0.999, 0.71, 0.06, 0.05],
    [0.001, 0.29, 0.94, 0.95]
], evidence=["B", "E"], evidence_card = [2, 2])

J = TabularCPD(variable = "J", variable_card = 2, values = [
    [0.95, 0.10],
    [0.05, 0.90]
], evidence = ['A'], evidence_card = [2])

M = TabularCPD(variable = "M", variable_card = 2, values = [
    [0.01, 0.70],
    [0.99, 0.30]
], evidence = ['A'], evidence_card = [2])

model.add_cpds(B, E, A, J, M)

# Checking if the cpds are valid for the model.
model.check_model()

# If John calls, predict the probability of burglary.
res = belief_propagation.query(variables=["B"], evidence = {"J":1})
print(res)
# If a burglary happened, predict the probability of John calling.
res = belief_propagation.query(variables=["J"], evidence = {"B":1})
print(res)


# Factor Graphs and Sum Product algorithm 
model = MarkovNetwork()
model.add_nodes_from(["W", "D", "S", "J", "T"])
edges = [
    ("W", "S"),
    ("S", "T"),
    ("S", "J"),
    ("D", "T"),
    ("D","S")
]

model.add_edges_from(edges)

# Add edges between the parents of the required nodes
# Connect nodes that should be moralized

model = model.triangulate()


# Define probabilities W (for Winter) and D (for Drunk). 
# You can use numpy arrays for the definition.
# Define conditional probabilities 
# WS (for Slick road given Winter), 
# SJ (for Jerry having an accident given slick road), 
# DST (for Tom having an accident given he is drunk and road is slick).
# You can use numpy arrays for the definition
# Note: W(0) denotes P(W=True) and W(1) denotes P(W=False)
W = np.array([0.5, 0.5])
D = np.array([0.6, 0.4])
WS = np.array([0.7, 0.3, 0.1, 0.9])
SJ = np.array([0.5, 0.5, 0.1, 0.9])
DST = np.array([0.9, 0.1, 0.5, 0.5, 0.5, 0.5, 0.1, 0.9])

# Define the factor nodes in the graph and add the factor potentials. 
# You can use the DiscreteFactor method from pgmpy for this.
factorA = DiscreteFactor(["W"], [2], W)
factorB = DiscreteFactor(["D"], [2], D)
factorC = DiscreteFactor(["W", "S"], [2, 2], WS.reshape(2,2))
factorD = DiscreteFactor(['S', 'J'], [2, 2], SJ.reshape(2,2))
factorE = DiscreteFactor(['D', 'S', 'T'], [2, 2, 2], DST.reshape(2,2,2))

# Add the defined factors into the graph.
model.add_factors(factorA, factorB, factorC, factorD, factorE)
print(model.check_model())


belief_propagation = BeliefPropagation(model)

# Question 2A
# What is the probability of road being slick if we have no evidence?
res1 = belief_propagation.query(variables=["S"])
print(res1)

# Question 2B
# If Jerry has an accident, find the marginal distribution of the road to be slick.
res2 = belief_propagation.query(variables=["S"], evidence={"J": 1})
print(res2)