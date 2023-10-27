from chromosome import Chromosome
import random


class Generation():
    def __init__(self,
                 fit_survival_rate: float,
                 unfit_survival_rate: float,
                 mutation_rate: float,
                 pop_size: int,
                 phase: int,
                 search_space: dict,
                 prev_best: Chromosome,
                 train_loader,
                 test_loader):
        self.fit_survival_rate = fit_survival_rate
        self.unfit_survival_rate = unfit_survival_rate
        self.mutation_rate = mutation_rate
        self.pop_size = pop_size
        self.phase = phase
        self.pop = []

        for i in range(pop_size):
            self.pop.append(Chromosome(phase=phase,
                                       prev_best=prev_best,
                                       genes=self.make_gene(search_space),
                                       train_loader = train_loader,
                                       test_loader = test_loader))

    def make_gene(self, search_space: dict):
        gene = {}
        keys = search_space.keys()
        for key in keys:
            gene[key] = random.choice(search_space[key])
        if self.phase == 0:
            gene['include_b'] = True
        return gene

    def sort_pop(self):
        sorted_pop = sorted(self.pop,
                            key=lambda x: x.fitness,
                            reverse=True)
        self.pop = sorted_pop

    def generate(self):
        print("start gen")
        self.sort_pop()
        # print(f"{[i.fitness for i in self.pop]}")
        num_fit_selected = int(self.fit_survival_rate * self.pop_size)
        num_unfit_selected = int(self.unfit_survival_rate * self.pop_size)
        num_mutate = int(self.mutation_rate * self.pop_size)

        new_pop = []

        for i in range(num_fit_selected):
            new_pop.append(self.pop[i])

        # print('ok')

        for i in range(num_unfit_selected):
            new_pop.append(self.pop[self.pop_size - i - 1])

        if (num_mutate > len(new_pop)):
            indices_to_mutate = random.sample(
                range(0, len(new_pop)), len(new_pop))
        else:
            indices_to_mutate = random.sample(
                range(0, len(new_pop)), num_mutate)
        
        for i in indices_to_mutate:
            new_pop[i] = new_pop[i].mutation()

        print("after fit/unfit/mutate", [i.fitness for i in new_pop])

        parents_list = []
        for i in range(self.pop_size - len(new_pop)):
            parents = random.sample(range(0, len(new_pop)), 2)
            parents_list.append(tuple(parents))

        for p1, p2 in parents_list:
            new_pop.append(new_pop[p1].crossover(new_pop[p2]))

        self.pop = new_pop
        self.pop_size = len(new_pop)
        self.sort_pop()
        # print(f"{[i.fitness for i in self.pop]}")

    def find_fittest(self):
        self.sort_pop()
        return self.pop[0]
