import numpy as np
from qfinance.option_pricing import OptionPricing, OptionParams

## define options
T = 40/365
myOptionParams = OptionParams(num_uncertainty_qubits=2, option_type="best-of-call")
myOptionParams.add_variable(
    {
    'S': 2.0,  
    'vol': 0.4 , 
    "r": 0.04,  
    "T": 40 / 365 , 
    }
)
myOptionParams.add_variable(
    {
    'S': 2.0,  
    'vol': 0.4 , 
    "r": 0.04,  
    "T": 40 / 365 , 
    }
)
myOptionParams.add_variable(
    {
    'S': 2.0,  
    'vol': 0.4 , 
    "r": 0.04,  
    "T": 40 / 365 , 
    }
)

correlation = np.array([[0.16*T, 0.032*T, 0.016*T], [0.032*T, 0.16*T, 0], [0.016*T, 0, 0.16*T]])

myOptionParams.set_covariance_matrix(correlation)
myOptionParams.set_strike_prices([2.01,1.9,1.8])

myOptionPricer = OptionPricing(myOptionParams)

estimate, conf_int = myOptionPricer.estimate_expectation(epsilon=0.001)
exact_expectation = myOptionPricer.compute_exact_expectation()
print(f"Estimated payoff: {estimate}")
print(f"Exact payoff: {exact_expectation}")
print(f"Confidence interval: {conf_int}")