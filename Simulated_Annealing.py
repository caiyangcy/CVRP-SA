import numpy as np
import time
import random

class SA:
    class Vehicle:
        def __init__(self, capacity):
            self.CAPACITY = capacity
            
    def __init__(self, customer, demand, init_temperature=5000, reduction_multiplier=0.99, iteration_multiplier=1.05, update_gap=5, max_time, vehicles, distance_callback):
        '''
        Args:
            customer: A dictionary mapping from customer label to the corresponding location. Include depot.
            demand: An 1d numpy array or list record the demand of each customer.
            init_temperature: The starting temperature.
            reduction_multiplier: Alpha in the second paper.
            iteration_multiplier: Beta in the second paper.
            update_gap: Time until next parameter update. M0 in the second paper.
            max_time: The maximum time the annealing process can take.
            vehicles: A list of available vehicles.
            distance_callback: A callable function to calculate the distance
        '''
        
        # Check types
        
        self.CUSTOMER = customer
        self.DEMAND = demand
        self.T = init_temperature
        self.ALPHA = reduction_multiplier
        self.BETA = iteration_multiplier
        self.M0 = update_gap
#        self.S0 = self._generate_init_solution()
        self.MAX_TIME = max_time
        
        total_capacity = sum([v.CAPACITY for v in vehicles])
        if total_capacity < np.sum(demand):
            raise ValueError("Total sum of capacity of all vehicles must be no less than the total demand")
        else:
            self.VEHICLES = vehicles
            
        if not callable(distance_callback):
            raise TypeError("distance_callback is not callable, should be method")
			
        self.distance_callback = distance_callback
    
    def _savings(self, method='Sequential'):
        return None
    
    def _split_solution(self, solution):
        vehicle_num = len(self.VEHICLES)
        solution_length = len(solution)-2 # Remove the first element and last element since they are both depot
        split_solutions = []
        current_solution = []
        vehicle_counter = 0
        solution_counter = 1 # Skip the depot
        curr_demand = 0
        current_vehicle = self.Vehicle[vehicle_counter]
        
        while vehicle_counter < vehicle_num and solution_counter < solution_length:
            current_solution += [solution[solution_counter]]
            curr_demand  += self.DEMAND[current_solution]
            if not (curr_demand < current_vehicle.CAPACITY):
                if curr_demand > current_vehicle.CAPACITY:
                    current_solution.pop()
                split_solutions.append([0]+current_solution+[0])
                vehicle_counter += 1
                current_solution = []
                current_vehicle = self.Vehicle[vehicle_counter]
                curr_demand = 0
            else:
                solution_counter += 1
                      
        return split_solutions
    
    def _generate_init_solution(self, method='random'):
        '''
        Args:
            method: Random or Savings. The method used to generate the inital solution. Available: random, savings_seq, savings_sara
                random: random permutations.
                savings: Clarke Wright Savings Algorithm. Sequential By default. 
        Return:
            A 2d list of solution(s)
        '''
        if method == 'random':
            customers = self.CUSTOMER.keys()
            init_solution = random.shuffle(customers)
            if init_solution[0] != 0:
                for i in range(len(init_solution)):
                    if init_solution[i] == 0:
                        init_solution[i] = init_solution[0]
            init_solution[0] = 0
            init_solution += [0]
            
            if np.sum(self.DEMAND) > self.VEHICLES[0].CAPACITY: # Need to split solutions
                return self._split_solution(init_solution)
            return init_solution
        
        return self._savings(method[8:])
    
    def _evaluate(self, solution):
        sum_distance = 0
        for idx in range(1, len(solution)):
            sum_distance += self.distance_callback(solution[idx], solution[idx-1])
        return sum_distance
    
    
    def _two_opt_exchange(self, solution, i, k):
        before = candidate_solution[:i]
        after = candidate_solution[k+1:]
        new_solution = before + candidate_solution[k:i-1:-1] + after
        
        return new_solution
    
    def _two_opt(self, solution, use_nearby=False):
        
        if use_nearby:
            nearby_20 = self._find_nearby_20(solution)
            return None
        
        best_solution = curr_solution
        best_distance = self._evaluate(curr_solution)
        eligible_nodes_num = len(curr_solution)
        for i in range(1, eligible_nodes_num-1):
            for k in  range(i+2, eligible_nodes_num):
                swapped_solution = self._two_opt_swap(curr_solution, i, k)
                swapped_distance = self._evaluate(swapped_solution)
                if swapped_distance < best_distance:
                    best_distance = swapped_distance
                    best_solution = swapped_solution
        return best_solution, best_distance
    
    def _insertion(self, solution, use_nearby=False):
        
        if use_nearby:
            nearby_20 = self._find_nearby_20(solution)
            return None
        
        return None
    
    def _find_nearby_20(self, curr_solution):
        nearby_point = 20
        if len(self.CUSTOMER) - len(curr_solution) < nearby_point:
            print('Cannot find 20 points nearby. Use all the rest points instead.')
            nearby_point = len(self.CUSTOMER) - len(curr_solution)
        
        
    def _local_search(self, curr_solution):
        '''
        This implementation is based on the first paper, where 2-opt exchange and insertion are used. 
        Probabilities also are used to get better solution.
        Args:
            curr_solution: Current solution.
        Return:
            Best solution during local search
        '''
        
        '''
        In the first paper, the probabilities for use 2-opt and insertion are 0.45, 0.45 respectively. 
        But here I'm going to make it half half.
        '''
        
        best_solution = None
        use_nearby_20 = [False, True]
        
        random_choices = np.random.random_sample((2,))
        use_nearby_20 = use_nearby_20[0] if random_choices[0] <= 0.95 else use_nearby_20[1]

        if random_choices[1] <= 0.5:
            return self._two_opt(curr_solution, use_nearby_20)
        
        return self._insertion(curr_solution, use_nearby_20)
    
    def _boltzmann(self):
        '''
        Use Metropolis criterion to generate a value between 0 and 1
        '''
        return 0
    
    def Simulated_Annealing(self, random_method='uniform'):
        '''
        The main algorithm
        '''
        start_time = time.time()
        alpha = self.alpha
        beta = self.beta
        M0 = self.M0
        T = self.T
        current_solution = self.S0
        current_cost = self._evaluate(current_solution)
        best_solution = self.S0
        best_cost = current_cost
        time = 0
        max_time = self.MAX_TIME # Haven't figured out what this variable is
        
        while time > max_time and T > 0.001:
            M = M0
            while M >= 0:
                new_solution = self._local_search(current_solution)
                new_cost = self._evaluate(new_solution)
                delta_cost = new_cost - current_cost
                if delta_cost < 0:
                    current_solution = new_solution
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_solution = current_solution
                        best_cost = current_cost
                else:
                    random = np.random.random_sample() if random_method == 'uniform' else self._boltzmann()
                    if random < (np.e)**(-delta_cost/T):
                        current_solution = new_solution
                        current_cost = new_cost
                M -= 1
            
            time += M0
            T *= alpha
            M0 *= beta
        
        end_time = time.time()
        print('\nFinished solving, with total time %s mins \n' % ((end_time - start_time)/60))
        return best_solution
        
        