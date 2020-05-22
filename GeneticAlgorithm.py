import sys
import random
import logging
import pandas as pd
from tabulate import tabulate
import copy

class test():
    def result(self):
        return random.randint(1,100)

class GeneticAlgorithm:
    def __init__(self, iteration= 50, population= 100, crossRate= 0.8, mutationRate= 0.2, eliteRate= 0.2):
        self.iteration = iteration
        self.current = 1
        self.population = population
        self.elements = dict()
        self.crossRate = crossRate
        self.mutationRate = mutationRate
        self.eliteRate = eliteRate
        self.best_chromosome = {'chromosome':dict(), 'fitness':0}

    def show_iteration(self): return self.iteration
    def show_population(self): return self.population
    def show_crossover(self): return self.crossRate
    def show_mutation(self): return self.mutation
    def show_elite(self): return self.elite

    def randomCont(self, minValue, maxValue):
        n = 0
        while n < self.iteration * self.population:
            yield random.random()*(maxValue-minValue)+minValue
            n += 1

    def randomDiscrete(self, Value):
        n = 0
        while n < self.iteration * self.population:
            yield random.choice(Value)
            n += 1

    def randomInt(self, Start, End):
        n = 0
        while n < self.iteration * self.population:
            yield random.randint(Start, End)
            n += 1

    def add_Elements(self, **kwargs):
        for k, v in kwargs.items():
            ### 連續數
            if len(v.split('-')) == 2 and ',' not in v and '.' not in v:
                try:
                    num = [int(value) for value in v.split('-')]
                    minV, maxV = min(num), max(num)
                    self.elements[k] = {'_range': v, '_type': 'Continous', '_generator': self.randomCont(minV, maxV)}
                except:
                    logging.error(f'Invalid Range: Please check the range of the element: {k}')
                    sys.exit()
            ### 離散型類型變數
            elif ',' in v:
                try:
                    self.elements[k] = {'_range': v, '_type': 'Discrete', '_generator': self.randomDiscrete(v.split(','))}
                except:
                    logging.error(f'Invalid Range: Please check the range of the element: {k}')
                    sys.exit()
            ### 離散型連續變數
            elif '...' in v:
                try:
                    num = [int(value) for value in v.split('...')]
                    minV, maxV = min(num), max(num)
                    self.elements[k] = {'_range': v, '_type': 'RandInt', '_generator': self.randomInt(minV, maxV)}
                except:
                    logging.error(f'Invalid Range: Please check the range of the element: {k}')
                    sys.exit()
        # self.elements = {k: self.elements[k] for k in sorted(self.elements.keys())}

    def remove_Elements(self, element):
        try:
            del self.elements[element]
            logging.info(f'Element: {element} is removed!')
        except:
            logging.info(f'Cannot find the element: {element}!')

    def show_Elements(self):
        logging.info(f'Module: Genetic Algorithm')
        logging.info(f'Element Number:　{len(self.elements.keys())}')
        df = pd.DataFrame(self.elements).T
        df = df.drop(['_generator'], axis=1)
        logging.info(f'\n{tabulate(df, headers="keys", tablefmt="fancy_grid")}')

    def Init_Population(self):
        self.chromosome = {index: {'chromosome': dict(), 'fitness': 0} for index in range(self.population)}
        if len(self.elements.keys()) == 0:
            logging.error('Please add some elements first!!')
            sys.exit()

        for index in range(self.population):
            self.chromosome[index]['chromosome'] = self.gen_Chromosome()

    def gen_Chromosome(self):
        chromosome = dict()
        for key, value in self.elements.items():
            chromosome[key] = next(value['_generator'])
        return chromosome

    def set_fitness(self, fitness):
        self.fitness = fitness
        logging.info('Fitness function is ready.')

    def calc_fitness(self):
        for key, value in self.chromosome.items():
            value['fitness'] = self.fitness.result(value)

    def show_Population(self, num= 10):
        if num > self.population or num <= 0:
            num = self.population
        logging.info('Current Population')
        title = f"{'Num':^5s}|{'|'.join([i.center(15) for i in self.elements.keys()])}|{' fitness':^15s}"
        print(title)
        print('-'*len(title))
        for index in range(num):
            print(f"{index:^5d}|{'|'.join([str(value)[:15].center(15) for value in self.chromosome[index]['chromosome'].values()])}|{str(self.chromosome[index]['fitness']):^15s}")

    def chromosome_Sorting(self):
        self.elite = sorted(self.chromosome.values(), key=lambda fit: fit['fitness'], reverse= True)
        for index in self.chromosome.keys():
            self.chromosome[index] = self.elite[index]

    def get_elite(self):
        self.elite = list()
        if self.best_chromosome['fitness'] > self.chromosome[self.population * self.eliteRate]['fitness']:
            self.elite.append(self.best_chromosome)
        if self.best_chromosome['fitness'] <= self.chromosome[0]['fitness']:
            self.best_chromosome = copy.deepcopy(self.chromosome[0])
        self.best_chromosome['iteration'] = self.current
        logging.info(f'Global Optimal: {self.best_chromosome}')
        index = 0
        while len(self.elite) < self.population * self.eliteRate:
            self.elite.append(self.chromosome[index])
            index += 1

    def next_population(self):
        for index in range(self.population):
            crossoverFlag = False
            mutationFlag = False
            ### 任選兩條菁英染色體
            chromosome1 = random.randint(0,(self.eliteRate*self.population)-1)
            chromosome2 = random.randint(0,(self.eliteRate*self.population)-1)
            while chromosome2 == chromosome1:
                chromosome2 = random.randint(0, (self.eliteRate * self.population) - 1)

            ### Crossover
            new_chromosome = {'chromosome': copy.deepcopy(self.elite[chromosome1]['chromosome']), 'fitness': 0}
            for key in self.elements:
                if random.random() < self.crossRate:
                    crossoverFlag = True
                    new_chromosome['chromosome'][key] = copy.deepcopy(self.elite[chromosome2]['chromosome'][key])
                if random.random() < self.mutationRate:
                    mutationFlag = True
                    new_chromosome['chromosome'][key] = next(self.elements[key]['_generator'])
            if not crossoverFlag and not mutationFlag:
                continue
            self.chromosome[index]['chromosome'] = new_chromosome['chromosome']
            self.chromosome[index]['fitness'] = 0

    def next_iteration(self):
        for iteration in range(self.iteration):
            self.calc_fitness()
            self.chromosome_Sorting()
            # self.show_Population()
            self.get_elite()
            self.next_population()
            self.current += 1
            # self.show_Population()
        return self.best_chromosome

if __name__ == "__main__":
    ### Log level
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(module)s - %(message)s'))
    console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(console_handler)

    a = GeneticAlgorithm()
    ## 加基因參數, a 為 0-1 之間連續數, b 為 x, y, z (離散型類型變數), c 為 0, 1, 2, ..., 10 (離散型連續變數)
    a.add_Elements(a='1-0', b='x,y,z', c='0...10')
    a.add_Elements(e='1-3', f='0,1', g='0...10')
    # a.removeElements('a')
    a.show_Elements()
    a.set_fitness(test)
    a.Init_Population()
    a.next_iteration()

