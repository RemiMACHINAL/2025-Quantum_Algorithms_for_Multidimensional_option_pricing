import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit_aer.primitives import Sampler
from qiskit_finance.circuit.library import LogNormalDistribution
from qfinance.ModifiedIQAE.mod_iae_updated import ModifiedIterativeAmplitudeEstimation
from qfinance.ModifiedIQAE.amplitude_estimator import AmplitudeEstimatorResult
from typing import List, Union, Tuple, Callable
from .qArithmetic import build_QNMSub, subtractorDraper
from .helper import loadNumber

class OptionParams:
    def __init__(self, num_uncertainty_qubits: int, option_type):
        self.individual_params = [] #est une liste de dictionaires chacun contenant : mu, sigma, mean, std, low, high
        self.cov = None #nulle ou matrices 3x3
        self.num_uncertainty_qubits = num_uncertainty_qubits
        self.strike_prices = None   #reel ou tableau de reel
        self.option_type = option_type

    def __str__(self):
        return str(
            {
                "individual_params": self.individual_params,
                "cov": self.cov,
                "num_uncertainty_qubits": self.num_uncertainty_qubits,
                "strike_prices": self.strike_prices,
                "option_type": self.option_type,
            }
        )

    def set_strike_prices(self, strike_prices: Union[List[float], float]):
        if isinstance(strike_prices, float):
            self.strike_prices = [strike_prices]
        else:
            self.strike_prices = strike_prices


    def set_covariance_matrix(self, correlation: Union[float, list]):
        variables = self.individual_params
        if len(variables) == 1:
            raise Exception("Covariance matrix not needed for single variable!")
        else:
            self.cov = correlation

    def add_variable(self, variable: Union[list, dict]): #prend un dictionnaire ou une liste de dictionnaire
        if type(variable) is dict:
            variable = self._process_variable(variable)
            self.individual_params.append(variable)
        elif type(variable) is list:
            for var in variable:
                curr_variable = self._process_variable(var)
                self.individual_params.append(curr_variable)

    def _process_variable(self, variable: dict):
        mu = (variable["r"] - 0.5 * variable["vol"] ** 2) * variable["T"] + np.log(
            variable["S"]
        )
        sigma = variable["vol"] * np.sqrt(variable["T"])
        mean = np.exp(mu + sigma**2 / 2)
        std = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2))
        if self.option_type == "spread call":
            low = 0
        else:
            low = np.maximum(0, mean - 3 * std)
        high = mean + 3 * std
        variable["mu"] = mu
        variable["sigma"] = sigma
        variable["mean"] = mean
        variable["std"] = std
        variable["low"] = low
        variable["high"] = high
        return variable




class OptionPricing:

    #
    # INITIALISATION
    #
    
    def __init__(self, options_params: OptionParams) -> None:
        self.num_variables = len(options_params.individual_params) #dimensions des sous jacents
        self.num_uncertainty_qubits = options_params.num_uncertainty_qubits
        self.options_params = options_params
        self.objective_index = None
        self.option = None #contiendra le circuit qui prépare le pay-off sur la distribution log-normale
        self.objective = None
        self.strike_price = options_params.strike_prices
        self.discount_factor = np.exp(-options_params.individual_params[0]["r"]*options_params.individual_params[0]["T"])
        self.define_uncertainty_model(options_params) #circuit avec les probas log-normale
        self.option_type = options_params.option_type



    #prepare la distribution log- normale
    def define_uncertainty_model(self, option_params: OptionParams) -> None:
        self.all_variables = option_params.individual_params
        self.dimension = len(self.all_variables)
        if self.dimension == 1:
            lower_bound = self.all_variables[0]["low"]
            upper_bound = self.all_variables[0]["high"]
            mu = self.all_variables[0]["mu"]
            std = self.all_variables[0]["sigma"]
            self.uncertainty_model = LogNormalDistribution(
                num_qubits=self.num_uncertainty_qubits,
                mu=mu,
                sigma=sigma**2,
                bounds=(lower_bound, upper_bound),
            )
        else:
            self.num_dist_qubits = [self.num_uncertainty_qubits] * self.dimension
            lower_bound = np.array([var["low"] for var in self.all_variables])
            upper_bound = np.array([var["high"] for var in self.all_variables])
            lower_bound = [np.min(lower_bound)] * self.dimension #pas sur de garder cette uniformisation
            upper_bound = [np.max(upper_bound)] * self.dimension # pareil
            mu = np.array([var["mu"] for var in self.all_variables])
            if option_params.cov is None:
                cov = np.diag([var["sigma"] ** 2 for var in self.all_variables])
            else:
                cov = option_params.cov
            self.uncertainty_model = LogNormalDistribution(
                num_qubits=self.num_dist_qubits,
                mu=mu,
                sigma=cov,
                bounds=list(zip(lower_bound, upper_bound)),
            )

        self.upper_bound = upper_bound
        self.lower_bound = lower_bound



    #
    # FONCTION RELATIVES AU PAY OFF
    #


    def _define_payoff_function(
        self, option_type: str, strike_price: list[float], c_approx=0.01
    ) -> None:
        if option_type == "call":
            strike_price = strike_price[0]
            self._define_basic_call_options(strike_price, c_approx)
        elif option_type == "basket call":
            strike_price = strike_price[0]
            self._define_basket_call_options(strike_price, c_approx)
        elif option_type == "spread call":
            strike_price = strike_price[0]
            self._define_spread_call_options(strike_price, c_approx)
        elif option_type == "call-on-max":
            strike_price = strike_price[0]
            self._define_call_on_max_options(strike_price, c_approx)
        elif option_type == "call-on-min":
            strike_price = strike_price[0]
            self._define_call_on_min_options(strike_price, c_approx)
        elif option_type == "best-of-call":
            self._define_best_of_call_options(strike_price, c_approx)
        else:
            raise Exception("Option type not defined!")



    def _define_best_of_call_options(self, strike_prices: list[float], c_approx=0.01) -> None:
        optionConstructor = BestOfCallPricer(self.num_uncertainty_qubits)
        circuit, post_processor = optionConstructor.construct_circuit(
            strike_prices,
            self.uncertainty_model,
            self.upper_bound,
            self.lower_bound,
            c_approx=c_approx,
        )
        self.option = circuit
        self.objective_index = circuit.num_qubits - 1
        self.post_processor = post_processor
        #print("draw")
        #circuit.draw(output="mpl", filename="circuit.png")


    #
    # EXACT PRICE
    #

    def _compute_exact_best_of_call_expectation(self) -> float:
        strike_price_1, strike_price_2, strike_price_3 = self.strike_price
        curr_exact_expectation = 0
        for i in range(len(self.uncertainty_model.probabilities)):
            curr_exact_expectation += self.uncertainty_model.probabilities[i] * max(
                0,
                self.uncertainty_model.values[i][0] - strike_price_1,
                self.uncertainty_model.values[i][1] - strike_price_2,
                self.uncertainty_model.values[i][2] - strike_price_3
            )
        return curr_exact_expectation


    def compute_exact_expectation(self) -> float:
        if self.option_type == "call":
            exact_value = self._compute_exact_call_expectation()
        elif self.option_type == "basket call":
            exact_value = self._compute_exact_basket_call_expectation()
        elif self.option_type == "spread call":
            exact_value = self._compute_exact_spread_call_expectation()
        elif self.option_type == "call-on-max":
            exact_value = self._compute_exact_call_on_max_expectation()
        elif self.option_type == "call-on-min":
            exact_value = self._compute_exact_call_on_min_expectation()
        elif self.option_type == "best-of-call":
            exact_value = self._compute_exact_best_of_call_expectation()
        else:
            raise Exception("Option type not defined!")
        # apply discount
        exact_value *= self.discount_factor
        return exact_value



    #
    # EXECUTION DU PROBLEME
    #

    def create_estimation_problem(self, epsilon=0.01) -> EstimationProblem:
        scaling_param = np.sqrt(epsilon)
        self._define_payoff_function(
            self.options_params.option_type,
            self.options_params.strike_prices,
            c_approx=scaling_param,
        )
        self.option_type = self.options_params.option_type
        # check if pre-requisites are satisfied
        if self.objective_index is None:
            raise Exception("Objective index not defined yet!")
        if self.option is None:
            raise Exception("Option not defined yet!")
        if self.post_processor is None:
            raise Exception("Post processor not defined yet!")

        # create estimation problem
        self.problem = EstimationProblem(
            state_preparation=self.option,
            objective_qubits=[self.objective_index],
            post_processing=self.post_processor,
        )
        return self.problem

    def run(
        self, epsilon=0.01, alpha=0.05, shots=100, method="MIQAE"
    ) -> AmplitudeEstimatorResult:
        # construct amplitude estimation
        if method == "IQAE":
            ae = IterativeAmplitudeEstimation(
                epsilon_target=epsilon,
                alpha=alpha,
                sampler=Sampler(run_options={"shots": shots}),
            )
            self.result = ae.estimate(self.problem)
        elif method == "MIQAE":
            ae = ModifiedIterativeAmplitudeEstimation(
                epsilon_target=epsilon,
                alpha=alpha,
                sampler=Sampler(run_options={"shots": shots}),
            )
            self.result = ae.estimate(self.problem, shots=shots)
        return self.result

    def process_results(self) -> Tuple[float, np.ndarray]:
        conf_int = np.array(self.result.confidence_interval_processed)
        estimated_value = self.result.estimation_processed
        # apply discount
        estimated_value *= self.discount_factor
        conf_int *= self.discount_factor
        return estimated_value, conf_int


    def estimate_expectation(
        self, epsilon=0.01, alpha=0.05, shots=100
    ) -> Tuple[float, np.ndarray]:
        self.create_estimation_problem(epsilon)
        self.run(epsilon, alpha, shots)
        return self.process_results()


class BestOfCallPricer:
    def __init__(self, num_uncertainty_qubits: int):
        self.num_uncertainty_qubits = num_uncertainty_qubits

    def _map_strike_price_int(
        self, strike_price: float, high: float, low: float, num_uncertainty_qubits: int
    ) -> float:
        return int(
            np.ceil(
                (strike_price - low) / (high - low) * (2**num_uncertainty_qubits - 1)
            )
        )

    def _map_strike_price_float(
        self, strike_price: float, high: float, low: float, num_uncertainty_qubits: int
    ) -> float:
        return (strike_price - low) / (high - low) * (2**num_uncertainty_qubits - 1)

    

    #encode le pay off des 3 calls individuellement et retourne les 3 circuits
    def objective_function_three_strike(
        self,
        num_uncertainty_qubits: int,
        strike_price_1: float,
        strike_price_2: float,
        strike_price_3: float,
        high: List[float],
        low: List[float],
        c_approx=0.01,
    ) -> Tuple[Gate, Gate, Gate, Callable, float]:
        
        # map strike prices
        mapped_strike_price_1_float = self._map_strike_price_float(
            strike_price_1, high[0], low[0], num_uncertainty_qubits
        )
        mapped_strike_price_1_int = self._map_strike_price_int(
            strike_price_1, high[0], low[0], num_uncertainty_qubits
        )

        mapped_strike_price_2_float = self._map_strike_price_float(
            strike_price_2, high[0], low[0], num_uncertainty_qubits
        )
        mapped_strike_price_2_int = self._map_strike_price_int(
            strike_price_2, high[0], low[0], num_uncertainty_qubits
        )

        mapped_strike_price_3_float = self._map_strike_price_float(
            strike_price_3, high[0], low[0], num_uncertainty_qubits
        )
        mapped_strike_price_3_int = self._map_strike_price_int(
            strike_price_3, high[0], low[0], num_uncertainty_qubits
        )

        step = (np.pi / 8) / (2**num_uncertainty_qubits - 1) * c_approx
        # determine offset
        offset_1 = (2**num_uncertainty_qubits - 1 - (mapped_strike_price_1_int - 1)) / 2
        offset_2 = (2**num_uncertainty_qubits - 1 - (mapped_strike_price_2_int - 1)) / 2
        offset_3 = (2**num_uncertainty_qubits - 1 - (mapped_strike_price_3_int - 1)) / 2

        offset = np.mean([offset_1, offset_2, offset_3])
        offset_angle = step * offset * 2

        # create first payoff function
        model_register = QuantumRegister(num_uncertainty_qubits, "model")
        objective_register = QuantumRegister(1, "objective")

        circuit_one = QuantumCircuit(model_register, objective_register)
        circuit_one.ry(-step * mapped_strike_price_1_float * 2, objective_register[0])
        for i in range(num_uncertainty_qubits):
            circuit_one.cry(
                step * 2 ** (i + 1), model_register[i], objective_register[0]
            )

        # Create second_payoff_function
        circuit_two = QuantumCircuit(model_register, objective_register)
        circuit_two.ry(-step * mapped_strike_price_2_float * 2, objective_register[0])
        for i in range(num_uncertainty_qubits):
            circuit_two.cry(
                step * 2 ** (i + 1), model_register[i], objective_register[0]
            )

        # Create third_payoff_function
        circuit_three = QuantumCircuit(model_register, objective_register)
        circuit_three.ry(-step * mapped_strike_price_3_float * 2, objective_register[0])
        for i in range(num_uncertainty_qubits):
            circuit_three.cry(
                step * 2 ** (i + 1), model_register[i], objective_register[0]
            )

        def post_processor(prob_of_one):
            coeff = (prob_of_one - 0.5) / (step) + offset
            fmax = high[0] - min(strike_price_1, strike_price_2, strike_price_3)
            og_range = (
                2**num_uncertainty_qubits
                - 1
                - min(mapped_strike_price_1_float, mapped_strike_price_2_float, mapped_strike_price_3_float)
            )
            return coeff / og_range * (fmax)

        return (
            circuit_one.to_gate(label="F1"),
            circuit_two.to_gate(label="F2"),
            circuit_three.to_gate(label="F3"),
            post_processor,
            offset_angle,
        )

    
    def construct_circuit(
        self,
        strike_prices: List[float],
        uncertainty_model: LogNormalDistribution, 
        high: List[float],
        low: List[float],
        c_approx=0.01,
    ) -> Tuple[QuantumCircuit, Callable]:

        # CONSTRUCTION DU CIRCUIT
        #S1
        first_var_register = QuantumRegister(self.num_uncertainty_qubits, "var1")
        #K1
        first_ancilla_register = QuantumRegister(self.num_uncertainty_qubits, "ancilla1")
        #contiendra 0 si S1 >= K1
        first_carry_register = QuantumRegister(1, "carry1")
        #copie car modifé dans build_QNMSub
        first_carry_register_copy = QuantumRegister(1, "carry1_copy")

        #Le prix S2
        second_var_register = QuantumRegister(self.num_uncertainty_qubits, "var2")
        #le strike K2
        second_ancilla_register = QuantumRegister(self.num_uncertainty_qubits, "ancilla2")
        second_ancilla_register_copy = QuantumRegister(self.num_uncertainty_qubits, "ancilla2_copy")
        #contiendra 0 si S2 >= K2
        second_carry_register = QuantumRegister(1, "carry2")
        #copie car modifé dans build_QNMSub 2x
        second_carry_register_copy = QuantumRegister(1, "carry2_copy")
        second_carry_register_copy_copy = QuantumRegister(1, "carry2_copy_copy")

        #Le prix S3
        third_var_register = QuantumRegister(self.num_uncertainty_qubits, "var3")
        #le strike K3
        third_ancilla_register = QuantumRegister(self.num_uncertainty_qubits, "ancilla3")
        #contiendra 0 si S3 >= K3
        third_carry_register = QuantumRegister(1, "carry3")

        #contiendra 1 si payoff1 > payoff2
        comparator_carry_1 = QuantumRegister(1, "comparator_carry_1")
        #contiendra 1 si payoff3 > max(payoff1, payoff2)
        comparator_carry_2 = QuantumRegister(1, "comparator_carry_2")

        # Qubit final pour marquer la fonction payoff
        objective_register = QuantumRegister(1, "objective")

        circuit = QuantumCircuit(
            third_var_register,
            third_ancilla_register,
            third_carry_register,
            first_var_register,
            first_ancilla_register,
            first_carry_register,
            first_carry_register_copy,
            second_var_register,
            second_ancilla_register,
            second_carry_register,
            second_ancilla_register_copy,
            second_carry_register_copy,
            second_carry_register_copy_copy,
            comparator_carry_1,
            comparator_carry_2,
            objective_register,
        )

        #on recupere les portes de payoff
        objective_1, objective_2, objective_3, post_processor, offset_angle = (
            self.objective_function_three_strike(
                self.num_uncertainty_qubits,
                strike_prices[0],
                strike_prices[1],
                strike_prices[2],
                high,
                low,
                c_approx,
            )
        )

        # ce circuit agit sur num_uncertainty_qubits + 1, num_uncertainty_qubits + 1, 1(comparator_carry) qbit
        #prend a et b, met le résultat de b-a dans b et comparator_carry vaudra 1 si a>b
        comparator = build_QNMSub( #dans qfinance.qArithmetics
            self.num_uncertainty_qubits + 1, self.num_uncertainty_qubits + 1
        ).to_gate()

        # permettra de comparer max(S1-K1, S2-K2) et S3-K3 si max est S2-K2
        controlled_comparator_1 = comparator.control(num_ctrl_qubits=1, ctrl_state="0")
        # permettra de comparer max(S1-K1, S2-K2) et S3-K3 si max est S1-K1
        controlled_comparator_2 = comparator.control(num_ctrl_qubits=1, ctrl_state="1")

        #ce circuit agit sur num_uncertainty_qubits, num_uncertainty_qubits, 1 (carry)
        #le second registre contenait b et contiendra le résultat de a-b
        #carry contient 0 si S-K>=0
        subtractor = subtractorDraper(self.num_uncertainty_qubits)

        #applique la porte objective 1 (ou 2 ou 3) que si les 3 qbit de controle sont à 110
        #un controle sur 0 verifie que S-K >= 0 et les autres verifient lequel des 3 payoff est superieur
        #attention comme il faut lire a l'envers, dans le circuit il faut d'abord indiquer le carry bit puis le comparator_ancilla
        #si comp carry1 = 1 et comp carry2 = 0
        controlled_objective_1 = objective_1.control(num_ctrl_qubits=3, ctrl_state="100")
        #si comp carry1 = 0 et comp carry2 = 0
        controlled_objective_2 = objective_2.control(num_ctrl_qubits=3, ctrl_state="000")
        #si comp carry2 = 1
        controlled_objective_3 = objective_3.control(num_ctrl_qubits=2, ctrl_state="10")

        #implémentation de la distribution log normale
        circuit.append(uncertainty_model,  third_var_register[:] + first_var_register[:] + second_var_register[:])

        # le qbit donnant le résultat est mis en état neutre |+> 
        # set sin(x) to sin(x+pi/4)
        circuit.ry(np.pi / 2, objective_register)
        # offset
        circuit.ry(-offset_angle, objective_register[0])

       
        # load strike prices
        loadNumber(circuit,first_ancilla_register,
            self._map_strike_price_int(
                strike_prices[0], high[0], low[0], self.num_uncertainty_qubits))
        loadNumber(circuit,second_ancilla_register,
            self._map_strike_price_int(
                strike_prices[1], high[0], low[0], self.num_uncertainty_qubits))
        loadNumber(circuit,third_ancilla_register,
            self._map_strike_price_int(
                strike_prices[2], high[0], low[0], self.num_uncertainty_qubits))

        loadNumber(circuit,second_ancilla_register_copy,
            self._map_strike_price_int(
                strike_prices[1], high[0], low[0], self.num_uncertainty_qubits))

        


        #S1-K1 stocké dans first_ancilla_register
        circuit.append(
            subtractor,
            first_var_register[:]
            + first_ancilla_register[:]
            + [first_carry_register[0]],
        )#copie
        circuit.cx(first_carry_register[0], first_carry_register_copy[0])
        #S2-K2 stocké dans second_ancilla_register
        circuit.append(
            subtractor,
            second_var_register[:]
            + second_ancilla_register[:]
            + [second_carry_register[0]],
        )
        circuit.append(
            subtractor,
            second_var_register[:]
            + second_ancilla_register_copy[:]
            + [second_carry_register_copy[0]],
        )
        #copie
        circuit.cx(second_carry_register[0], second_carry_register_copy_copy[0])
        #S3-K3 stocké dans third_ancilla_register
        circuit.append(
            subtractor,
            third_var_register[:]
            + third_ancilla_register[:]
            + [third_carry_register[0]],
        )


                  
        #comparator_carry1 vaudra 1 si S2-K2 < S1-K1
        circuit.append(
            comparator,
            first_ancilla_register[:]
            + first_carry_register[:]
            + second_ancilla_register[:]
            + second_carry_register[:]
            + comparator_carry_1[:]
        )



        #si S2-K2 > S1-K1 on compare S2-K2 et S3-K3 ie comparator_carry_1 = 0
        circuit.append(
            controlled_comparator_1,
            [comparator_carry_1[0]]
            + third_ancilla_register[:]
            + third_carry_register[:]
            + second_ancilla_register_copy[:]
            + second_carry_register_copy[:]
            + comparator_carry_2[:]
        )



        #si S2-K2 < S1-K1 on compare S1-K1 et S3-K3 ie comparator_carry_1 = 1
        circuit.append(
            controlled_comparator_2,
            [comparator_carry_1[0]]
            + third_ancilla_register[:]
            + third_carry_register[:]
            + first_ancilla_register[:]
            + first_carry_register[:]
            + comparator_carry_2[:]
        )


        #Maintenant :
        #comparator_carry_1 = 0 si S2-K2 >= S1-K1
        #                     1 si S2-K2 < S1-K1
        #
        #comparator_carry_2 = 1 si S3-K3 > max
        #                   = 0 si S3-K3 =< max

        #si 0 et 0 et second_carry_register_copy=0
        circuit.append(
            controlled_objective_2,
            second_carry_register_copy_copy[:]
            + [comparator_carry_2[0]]
            + [comparator_carry_1[0]]
            + second_var_register[:]
            + objective_register[:],
        )


        #si 1 et 0 et first_carry_register_copy=0
        circuit.append(
            controlled_objective_1,
            first_carry_register_copy[:]
            + [comparator_carry_2[0]]
            + [comparator_carry_1[0]]
            + first_var_register[:]
            + objective_register[:],
        )


        #si comparator_carry2 = 1 et third_carry_register=0
        circuit.append(
            controlled_objective_3,
            third_carry_register[:]
            + [comparator_carry_2[0]]
            + third_var_register[:]
            + objective_register[:],
        )

        return circuit, post_processor
