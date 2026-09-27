"""
@Author: 
Konstantinos Konstantinou s1118596
Levente Bódi s1122062

on the basis of the files by

Joris van Vugt, Moira Berens, Leonieke van den Bulk

Entry point for the creation of the variable elimination algorithm in Python 3.
Code to read in Bayesian Networks has been provided. We assume you have installed the pandas package.

"""
"""

"""
from read_bayesnet import BayesNet
from variable_elim import VariableElimination


if __name__ == '__main__':
    # The class BayesNet represents a Bayesian network from a .bif file in several variables
    net = BayesNet('earthquake1.bif')  # Format and other networks can be found on http://www.bnlearn.com/bnrepository/

    # Print details of the Bayesian network
    """
    print("Nodes:")
    print(net.nodes)
    print("Values:")
    print(net.values)
    print("Parents:")
    print(net.parents)
    print("Probabilities:")
    print(net.probabilities)
    """

    query = 'Alarm'
    observed = {'Burglary': 'True'}
    elim_order = net.nodes
    elim_order.remove(query)
    ve = VariableElimination(net, log_file='variable_elimination_log.txt')

    # If elim_order is None it uses _contained_in_fewest_factors_first_heur
    print(ve.run(query = query, observed = observed, elim_order = None))
