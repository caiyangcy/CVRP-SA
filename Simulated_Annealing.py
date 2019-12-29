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
            customer: A dictionary mapping from customer label to the corresponding location. Include depot. locations are numpy arrays.
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
        
        # customer	
        if type(customer) is not dict:
            raise TypeError("Customers' type must be a dictionary")
    		
        if len(customer) == 0:
            raise ValueError("Customers are empty")
        
        self.CUSTOMER = customer
        
        # demand
        if type(demand) is not list or type(demand) is not np.ndarray:
            raise TypeError("Demand's type must be a list or numpy array")
    		
        if np.min(demand) < 0:
            raise ValueError("Demand cannot be negative")
        
        if len(demand) != len(customer):
            raise ValueError("Demand and customers do not have the same number of elements")
        
        self.DEMAND = demand
        
        # init_temperature
        if type(init_temperature) is not int or type(init_temperature) is not float:
            raise TypeError("Initial temperature must be int or float")
            
        if init_temperature < 0:
            raise ValueError('Initial temperature cannot be negative')
        
        self.T = init_temperature
        
        # alpha
        if type(reduction_multiplier) is not float:
            raise TypeError("Reduction multiplier must be int or float")
            
        if reduction_multiplier <= 0 or reduction_multiplier >= 1:
            raise ValueError('Reduction multiplier must be between 0 and 1')
            
        self.ALPHA = reduction_multiplier
        
        # beta
        if type(iteration_multiplier) is not float:
            raise TypeError("Iteration multiplier must be int or float")
            
        if reduction_multiplier <= 1:
            raise ValueError('Iteration multiplier must larger than 1')
            
        self.BETA = iteration_multiplier
        
        # M0
        if type(update_gap) is not int or type(update_gap) is not float:
            raise TypeError("Update gap must be int or float")
            
        if update_gap <= 0:
            raise ValueError('Update gap should be larger than 1')
        
        if type(update_gap) is float:
            print('Update gap is rounded up')
            update_gap = int(update_gap)+1
            
        self.M0 = update_gap
#        self.S0 = self._generate_init_solution()
        
#        self.MAX_TIME = max_time
        
        # capacity
        total_capacity = sum([v.CAPACITY for v in vehicles])
        if total_capacity < np.sum(demand):
            raise ValueError("Total sum of capacity of all vehicles must be no less than the total demand")
        else:
            self.VEHICLES = vehicles
            
        # distance callback
        if not callable(distance_callback):
            raise TypeError("distance_callback is not callable, should be method")
			
        self.distance_callback = distance_callback
    
    def _savings(self, method='Sequential'):
        return None
    
    def _split_solution(self, solution):
        '''
        Split the given solution which violates the capacity constraint into multiple solutions without violation
        '''
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
    
    def _evaluate(self, solutions):
        obj_value = 0
        for solution in solutions:
            obj_value += self.distance_callback(solution)
        return obj_value
    
    def _two_exchange(self, solution, use_nearby=False):
        
        if use_nearby:
            nearby_20 = self._find_nearby_20(solution)
            return None
        
        # Random pick up solution
        solution_number = len(solution)
        candidates = np.random.randint(low=0, high=solution_number,size=2)
        i, k = candidates[0], candidates[1]
        solution_1 = solution[i]
        solution_2 = solution[k]
        
        # Random pick up the points that are going to be swapped
        point_1 = np.random.choice(solution_1[1:-1], 1)[0]
        point_2 = np.random.choice(solution_2[1:-1], 1)[0]
                    
        new_solution_1 = solution_1[:point_1]+[point_2]+solution_1[point_1:]
        new_solution_2 = solution_2[:point_2]+[point_1]+solution_2[point_2:]
        
#        if self.VEHICLES[i].CAPACITY >= self.DEMAND[new_solution_1].sum() and self.VEHICLES[k].CAPACITY >= self.DEMAND[new_solution_2].sum():
        solution[i] = new_solution_1
        solution[k] = new_solution_2
        
        return solution
    
    def _fix_insertion_violation(self, i, k, new_solution):
        
        '''
        k : The solution being inserted a new element
        
        Fix the violation of capacity constraints after 2-opt or insertion
        
        Note that this is a very simple fixup.
        '''
        
        insert_solution = new_solution[i]
        insert_solution_demand = self.DEMAND[insert_solution]
        
        if insert_solution_demand > self.VEHICLES[k].CAPACITY:
            # Remove the last element
            last_element = insert_solution[-2]
            diff = insert_solution_demand - self.VEHICLES[k].CAPACITY
            fixed = False
            candidate_solutions_labels = np.arange(0, len(new_solution)).delete(i, k)
            for idx in candidate_solutions_labels:
                if diff <= self.VEHICLES[idx].CAPACITY - self.DEMAND[new_solution[idx]].sum() - self.DEMAND[last_element]:
                    # Insert it into a random position
                    position = np.random.choice(new_solution[idx][1:-1], 1)[0]
                    insert_solution = insert_solution[::-2]+insert_solution[-1]
                    
                    new_insert_solution = new_solution[idx]
                    new_insert_solution = new_insert_solution[:position] + [last_element] + new_insert_solution[position:]
                    
                    new_solution[k] = insert_solution
                    new_solution[idx] = new_insert_solution
                    fixed = True
                    break
        
        return [fixed, new_solution]
    
    def _insertion(self, solution, use_nearby=False):
        
        if use_nearby:
            nearby_20 = self._find_nearby_20(solution)
            return None
        
        # Pick two random solutions. Delete one point from one solution and insert it into the other
        
        # Keep a copy of the original solution for backup later
        curr_solution = solution.copy()
        
        solution_number = len(solution)
        candidates = np.random.randint(low=0, high=solution_number,size=2)
   
        remove_solution = solution[candidates[0]]
        insert_solution = solution[candidates[1]]
        
        removed_point = np.random.choice(remove_solution[1:-1], 1)[0]
        inserted_point = np.random.choice(insert_solution[1:-1], 1)[0]
        
        remove_solution = remove_solution[:removed_point]+remove_solutionp[remove_solution+1:]
        insert_solution = insert_solution[:inserted_point]+[removed_point]+insert_solution[inserted_point:]
        
        solution[candidates[0]] = remove_solution
        solution[candidates[1]] = insert_solution
        
#        results = self._fix_insertion_violation(candidates[0], candidates[1], solution)
#        
#        if results[0]:
#            return curr_solution
#        
#        return results[1]
        
        return solution
    
    def _find_nearby_20(self, curr_solution):
        '''
        Given the current solution, find the 20 nearest points to the random chosen solution.
        
        This function is not implemented yet since the actually process to achieve this might be time-consuming.
        
        KD-Tree might be a proper way to achieve this.
        '''
        solution_number = len(curr_solution)
        solution_to_be_inserted = curr_solution[np.random.randint(low=0, high=solution_number,size=1)[0]]
        
        nearby_point = 20
        if len(self.CUSTOMER) - len(solution_to_be_inserted) < nearby_point:
            print('Cannot find 20 points nearby. Use all the rest points instead.')
            nearby_point = len(self.CUSTOMER) - len(solution_to_be_inserted)
        
        
        
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
            return self._two_exchange(curr_solution, use_nearby_20)
        
        return self._insertion(curr_solution, use_nearby_20)
    
    def _boltzmann(self):
        '''
        Use Metropolis criterion to generate a value between 0 and 1
        '''
        return 0
    
    def _check_violation(self, solution):
        '''
        Return:
            True if no violation. False otherwise.
        '''
        for idx in range(len(solution)):
            sub_route = solution[idx]
            demand = self.DEMAND[sub_route].sum()
            capacity = self.VEHICLES[idx].CAPACITY
            if capacity < demand:
                return False
        return True
    
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
                if delta_cost < 0 and self._check_violation(new_solution): # Make sure there is no violation
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
        
        