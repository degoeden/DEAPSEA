from deap import base, creator, tools
import random
from concurrent.futures import ThreadPoolExecutor
import csv
import os
import numpy as np

class DeapSeaGa:
    def __init__(self, objective, BOUNDS, BITS, NPOP=10, CXPB=0.5, MUTPB=0.2, NGEN=40, ELITES_SIZE=1, TOURNAMENT_SIZE = 3, PATIENCE=None, TOL=1e-3, NWORKERS=1, csv_path=None):
        self.objective = objective
        self.BOUNDS = BOUNDS
        self.BITS = BITS
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.NGEN = NGEN
        self.NPOP = NPOP
        self.ELITES_SIZE = ELITES_SIZE
        self.TOURNAMENT_SIZE = TOURNAMENT_SIZE
        self.PATIENCE = PATIENCE
        self.TOL = TOL
        self.NWORKERS = NWORKERS
        self.csv_path = csv_path
        self.build_toolbox()

    def decode(self, ind):
        decoded = {}
        cursor = 0
        for key in self.BOUNDS.keys():
            b = self.BITS[key]
            bit_slice = ind[cursor:cursor + b]
            cursor += b
            intval = int("".join(str(x) for x in bit_slice), 2)
            max_int = 2**b - 1
            low, high = self.BOUNDS[key]
            val = low + (high - low) * intval / max_int
            decoded[key] = val
        return decoded

    def encode(self, design_dict):
        encoded_bits = []
        for key in self.BOUNDS.keys():
            low, high = self.BOUNDS[key]
            b = self.BITS[key]
            max_int = 2**b - 1
            val = max(min(design_dict[key], high), low)
            intval = int(np.round((val - low) / (high - low) * max_int))
            bitstring = [int(x) for x in f"{intval:0{b}b}"]
            encoded_bits.extend(bitstring)
        return creator.Individual(encoded_bits)

    def wrapped_objective(self, bit_ind):
        float_ind = self.decode(bit_ind)
        return self.objective(float_ind)

    def build_toolbox(self):
        creator.create("f", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.f)
        total_bits = sum(self.BITS.values())
        self.toolbox = base.Toolbox()
        def gen_bits():
            return [random.randint(0, 1) for _ in range(sum(self.BITS.values()))]
        self.toolbox.register("individual", tools.initIterate, creator.Individual, gen_bits)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.wrapped_objective)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/total_bits)
        self.toolbox.register("select", tools.selTournament, tournsize=self.TOURNAMENT_SIZE)

        if self.NWORKERS > 1:
            self.pool = ThreadPoolExecutor(max_workers=self.NWORKERS)
            self.toolbox.register("map", self.pool.map)
        else:
            self.toolbox.register("map", map)

    def evaluate_population(self, pop):
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
    
    def crossover(self, offspring):
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.CXPB:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        return offspring

    def mutate(self, offspring):
        for mutant in offspring:
            if random.random() < self.MUTPB:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values
        return offspring
    
    def log_generation(self, pop, generation):
        if self.csv_path is None:
            return
        
        decoded_sample = self.decode(pop[0])
        fieldnames = list(decoded_sample.keys()) + ['fitness', 'generation']
        file_exists = os.path.isfile(self.csv_path)

        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            for bit_ind in pop:
                decoded = self.decode(bit_ind)
                row = dict(decoded)
                row['fitness'] = bit_ind.fitness.values[0]
                row['generation'] = generation
                writer.writerow(row)

    def run(self):
        if self.csv_path and os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        pop = self.toolbox.population(n=self.NPOP)
        best_fitness = float('inf')

        # Evaluate the entire population
        self.evaluate_population(pop)

        for g in range(self.NGEN):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            offspring = self.crossover(offspring)
            offspring = self.mutate(offspring)

            # Evaluate the individuals with an invalid fitness
            self.evaluate_population(offspring)
            # Log the generation
            self.log_generation(offspring, g)
            
            # The population is replaced by the offspring, but keeps elites
            elites = tools.selBest(pop, self.ELITES_SIZE)
            offspring = tools.selBest(offspring, len(offspring) - self.ELITES_SIZE)
            pop[:] = elites + offspring

            # ---- EARLY STOPPING CHECK ----
            current_best = min(ind.fitness.values[0] for ind in pop)
            if abs((best_fitness - current_best)/(best_fitness)) < self.TOL:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                best_fitness = current_best

            if self.PATIENCE is not None and stagnation_counter > self.PATIENCE:
                print(f"Stopping early at generation {g+1} due to lack of patience.")
                break
            
        if hasattr(self, "pool"):
            self.pool.shutdown()

        if self.PATIENCE is None or stagnation_counter <= self.PATIENCE:
            print(f"Finished running all {self.NGEN} generations, might not have converged.")

        # Find and return the fittest individual
        fits = [ind.fitness.values[0] for ind in pop]
        best_index = fits.index(min(fits))
        decoded_best = self.decode(pop[best_index])
        return decoded_best, pop[best_index].fitness.values[0]
        