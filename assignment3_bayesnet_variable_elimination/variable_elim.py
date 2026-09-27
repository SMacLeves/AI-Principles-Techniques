"""
@Author: 
Konstantinos Konstantinou s1118596
Levente Bódi s1122062

on the basis of the files by

Joris van Vugt, Moira Berens, Leonieke van den Bulk

Class for the implementation of the variable elimination algorithm.

"""
import pandas as pd
from collections import Counter

class VariableElimination:
    """
    A class to perform the Variable Elimination algorithm on a Bayesian Network.

    This class provides a framework to calculate the probability distribution
    of a query variable given observed variables by eliminating irrelevant variables
    using the Variable Elimination algorithm.

    Attributes:
        network (object): The Bayesian Network instance.
        log_file (str): The path to the log file for recording steps and results.

    Methods:
        run(query, observed=None, elim_order=None):
            Executes the Variable Elimination algorithm to compute the probability distribution
            for the query variable given the observed variables and elimination order.

        _initialize_factors(observed):
            Initializes factors from the network's probability tables, applying observed evidence.

        _eliminate_variable(factors, variable):
            Eliminates a specified variable from the factors by summing it out.

        _combine_factors(factors):
            Multiplies a list of factors to produce a single combined factor.

        _sum_out_variable(factor, variable):
            Sums out a variable from a factor to marginalize it.

        _normalize(factor):
            Normalizes a factor to ensure it represents a valid probability distribution.
    """

    def __init__(self, network, log_file='ve_log.txt'):
        """
        Initializes the Variable Elimination instance.

        Args:
            network (object): The Bayesian Network instance.
            log_file (str, optional): The path to the log file for recording steps and results.
                                      Defaults to 've_log.txt'.
        """
        self.network = network
        self.log_file = log_file

        # Clear the log file at the start
        with open(self.log_file, 'w') as log:
            log.write("Variable Elimination Log\n")
            log.write("========================\n")

    def _log(self, message):
        """
        Writes a message to the log file.

        Args:
            message (str): The message to log.
        """
        with open(self.log_file, 'a') as log:
            log.write(message + "\n")

    def run(self, query, observed=None, elim_order=None):
        """
        Executes the Variable Elimination algorithm.

        Args:
            query (str): The query variable for which the probability distribution is to be computed.
            observed (dict, optional): A dictionary of observed variables and their values in the form {variable: value}.
                                        Defaults to None.
            elim_order (list or function, optional): A list specifying the elimination order, or a function that generates
                                                     the order based on the network. Defaults to None.

        Returns:
            DataFrame: A pandas DataFrame representing the normalized probability distribution for the query variable.
        """
        self._log(f"Query: {query}")
        self._log(f"Observed: {observed if observed else 'None'}")
        
        # Initialize factors and apply observed evidence if there is observed evidence
        factors = self._initialize_factors(observed)
        self._log(f"Initial factors: {len(factors)}")
        for factor in factors:
            self._log(f"Factor:\n{factor}")

        # Determine elimination order if not provided
        if elim_order is None:
            #elim_order = [var for var in self.network.nodes if var != query]
            elim_order = self._contained_in_fewest_factors_first_heur(factors)
        print(f"Elimination Order: {self._contained_in_fewest_factors_first_heur(factors)}")
        self._log(f"Elimination Order: {elim_order}")

        for var in elim_order:
            if var != query:
                self._log(f"Eliminating variable: {var}")
                factors = self._eliminate_variable(factors, var)
                self._log(f"Factors after eliminating {var}: {len(factors)}")
                for factor in factors:
                    self._log(f"Factor:\n{factor}")

        # Combine remaining factors
        query_factor = self._combine_factors(factors)
        self._log("Final combined factor before normalization:")
        self._log(str(query_factor))

        # Normalize the final factor
        normalized_query = self._normalize(query_factor)
        self._log("Final normalized factor:")
        self._log(str(normalized_query))

        return normalized_query

    def _initialize_factors(self, observed):
        """
        Initializes factors from the network's probability tables, applying evidence if provided.

        Args:
            observed (dict): A dictionary of observed variables and their values.

        Returns:
            list: A list of pandas DataFrames representing the factors.
        """
        factors = []
        for var, prob_table in self.network.probabilities.items():
            df = pd.DataFrame(prob_table)

            # Apply evidence only if observed is not None
            if observed:
                for obs_var, obs_val in observed.items():
                    if obs_var in df.columns:
                        # Filter rows matching the observed values
                        df = df[df[obs_var] == obs_val]
            
            if not df.empty:
                factors.append(df)  # Add non-empty factors
        print(factors)
        return factors
    def _contained_in_fewest_factors_first_heur(self, factors):
            """
            Determines an elimination order for variables based on their frequency 
            across factors, prioritizing those that appear in the fewest factors.

            Args:
                factors (list): A list of factors, each with a `columns` attribute 
                                (e.g., pandas DataFrame).

            Returns:
                list: A sorted list of unique variables by frequency, excluding 'prob'.
            """
            elim_order = []
            # Append the column names to a list elim_order of all factors
            for i in factors:
                elim_order.append(list(i.columns))
            # Flatten the list of lists to a single list with all values
            flattened_list = sum(elim_order, [])
            # Sort the list according to frequencies in an ascending order
            freq = Counter(flattened_list)
            sorted_list = sorted(flattened_list, key=lambda x: (freq[x], x))
            # Remove duplicates
            unique_list = []
            for item in sorted_list:
                if item not in unique_list:
                    unique_list.append(item)
            # Remove 'prob' from the list
            unique_list.remove('prob')
            return unique_list

    def _eliminate_variable(self, factors, variable):
        """
        Eliminates a specified variable from the factors by summing it out.

        Args:
            factors (list): A list of pandas DataFrames representing the factors.
            variable (str): The variable to eliminate.

        Returns:
            list: Updated list of factors after the variable has been eliminated.
        """
        # Create two lists, list of factors which include the factor/ variable, and one which does not
        factors_with_var = [factor for factor in factors if variable in factor.columns]
        factors_without_var = [factor for factor in factors if variable not in factor.columns]

        if not factors_with_var:
            self._log(f"No factors involve the variable {variable}. Skipping elimination.")
            return factors_without_var
        # Combine the factors that contain the variable/ factor
        combined_factor = self._combine_factors(factors_with_var)
        # Sum out the variable/ factor
        reduced_factor = self._sum_out_variable(combined_factor, variable)

        factors_without_var.append(reduced_factor)
        return factors_without_var

    def _combine_factors(self, factors):
        """
        Multiplies a list of factors to produce a single combined factor.

        Args:
            factors (list): A list of pandas DataFrames representing the factors.

        Returns:
            DataFrame: A pandas DataFrame representing the combined factor.
        """
        if not factors:
            raise ValueError("No factors provided for combination. Check your input data or elimination process.")
        # Initialize a starting factor
        merged_factor = factors[0]
        # Iterating over the other factors
        for factor in factors[1:]:
            # Identify the joint variables between merged_factor and the next factor
            merge_columns = [col for col in merged_factor.columns if col in factor.columns and col != 'prob']
            # Merge merging factor with factor on common variabes
            merged_factor = merged_factor.merge(factor, on=merge_columns, suffixes=('_left', '_right'))
            # Assign prob_left and prob_right to the two factors and multiply them
            merged_factor['prob'] = merged_factor['prob_left'] * merged_factor['prob_right']
            # Drop any uneccessary columns, such as prob_left, prob_right
            merged_factor = merged_factor.drop(columns=['prob_left', 'prob_right'])
        return merged_factor

    def _sum_out_variable(self, factor, variable):
        """
        Sums out a variable from a factor to marginalize it.

        Args:
            factor (DataFrame): A pandas DataFrame representing the factor.
            variable (str): The variable to sum out.

        Returns:
            DataFrame: A pandas DataFrame with the variable summed out.
        """
        # Group by the columns except the column to be summed out by first determining the columns and then performing the actual grouping
        group_by_columns = [col for col in factor.columns if col not in [variable, 'prob']]
        return factor.groupby(group_by_columns, as_index=False)['prob'].sum()

    def _normalize(self, factor):
        """
        Normalizes a factor to ensure it represents a valid probability distribution.

        Args:
            factor (DataFrame): A pandas DataFrame representing the factor.

        Returns:
            DataFrame: A normalized pandas DataFrame.
        """
        # Sum and divide to normalize the final factor
        total_prob = factor['prob'].sum()
        factor['prob'] = factor['prob'] / total_prob
        return factor





        




