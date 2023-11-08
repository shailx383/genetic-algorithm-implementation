import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
# from generation import Generation

SEARCH_SPACE = {
    'k_size_a': [1, 3, 5],
    'k_size_b': [1, 3, 5],
    'k_size_c': [1, 3, 5],
    'k_size_d': [1, 3, 5],
    'out_channels_a': [8, 16, 32, 64],
    'out_channels_b': [8, 16, 32, 64],
    'out_channels_c': [8, 16, 32, 64],
    'out_channels_d': [8, 16, 32, 64],
    'include_pool_a': [True, False],
    'include_pool_b': [True, False],
    'include_pool_c': [True, False],
    'include_pool_d': [True, False],
    'pool_type_a': ['max_pooling','avg_pooling'],
    'pool_type_b': ['max_pooling','avg_pooling'],
    'pool_type_c': ['max_pooling','avg_pooling'],
    'pool_type_d': ['max_pooling','avg_pooling'],
    'activation_type_a': ['relu', 'tanh', 'elu', 'selu'],
    'activation_type_b': ['relu', 'tanh', 'elu', 'selu'],
    'activation_type_c': ['relu', 'tanh', 'elu', 'selu'],
    'activation_type_d': ['relu', 'tanh', 'elu', 'selu'], 
    'include_b': [True, False],
    'include_c': [True, False],
    'include_d': [True, False],
    'include_BN_a': [True, False],
    'include_BN_b': [True, False],
    'include_BN_c': [True, False],
    'include_BN_d': [True, False],
    'skip_connection': [False],
}

FIT_SURVIVAL_RATE = 0.5
UNFIT_SURVIVAL_RATE = 0.2
MUTATION_RATE = 0.1
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
kwargs = {'batch_size': 64, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(trainset, **kwargs)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(testset, **kwargs)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import torch
import random
 
class FinalModel(nn.Module):
    def __init__(self, chromosome):
        super().__init__()
        self.block = chromosome.model
        if chromosome.phase == 0:
            in_channels = 3
            if(chromosome.genes['include_d']):
                out_channels = chromosome.genes['out_channels_d']
            elif(chromosome.genes['include_c']):
                out_channels = chromosome.genes['out_channels_c']
            elif(chromosome.genes['include_b']):
                out_channels = chromosome.genes['out_channels_b']
            else:
                out_channels = chromosome.genes['out_channels_a']
        else:
            if(chromosome.prev_best.genes['include_b']):
                in_channels = chromosome.prev_best.genes['out_channels_b']
            else:
                in_channels = chromosome.prev_best.genes['out_channels_a']
            if(chromosome.genes['include_b']):
                out_channels = chromosome.genes['out_channels_b']
            else:
                out_channels = chromosome.genes['out_channels_a']
        # self.skip = nn.Conv2d(in_channels, out_channels, 1)
 
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_channels*chromosome.out_dimensions**2,10)
 
    def forward(self, x, chromosome):
        # if chromosome.genes['skip_connection']:
        #     y = x
        #     if chromosome.phase != 0:
        #         y = chromosome.prev_best.model(x)
        #     y = self.skip(y)
        x=self.block(x)
        # if chromosome.genes['skip_connection']:
        #     x = x + y
        x = self.fc(self.flatten(x))
        x = F.log_softmax(x, dim=1)
        return x
 
 
class Chromosome:
    def __init__(self,phase:int,prev_best,genes:dict,train_loader,test_loader):
        self.phase = phase
        self.prev_best = prev_best
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.genes = genes
        self.out_dimensions = prev_best.out_dimensions if phase!=0 else 32
        self.fitness = -1 
        self.model:nn.Module = self.build_model()
        self.train_loader = train_loader
        self.test_loader = test_loader
        if self.fitness==-1:
            self.fitness = self.fitness_function(train_loader,test_loader)
 
    def build_model(self)->nn.Module:
        if(self.prev_best!=None):
            prev_best_model:nn.Module = self.prev_best.model
        new_model_modules = []
        padding_size = 0
        if(self.genes['skip_connection']):
            padding_size = 16 if self.phase==0 else self.prev_best.out_dimensions//2
        if(self.out_dimensions<self.genes['k_size_a']):
            self.fitness = 0
            return nn.Sequential()
        
        if(self.phase!=0):
            # layer_a = nn.Conv2d(self.prev_best.genes['out_channels_b'] if self.prev_best.genes['include_b'] else self.prev_best.genes['out_channels_a'],self.genes['out_channels_a'],self.genes['k_size_a'],padding = self.genes['k_size_a']//2 if self.genes['skip_connection'] else 0)
            layer_a = nn.Conv2d(self.prev_best.genes['out_channels_b'] if self.prev_best.genes['include_b'] else self.prev_best.genes['out_channels_a'],self.genes['out_channels_a'],self.genes['k_size_a'],padding = 'same')
        else:
            # layer_a = nn.Conv2d(3,self.genes['out_channels_a'],self.genes['k_size_a'],padding = self.genes['k_size_a']//2 if self.genes['skip_connection'] else 0)
            layer_a = nn.Conv2d(3,self.genes['out_channels_a'],self.genes['k_size_a'],padding = 'same')
        # self.out_dimensions = (self.out_dimensions-self.genes['k_size_a']+1)
        new_model_modules.append(layer_a)
        if(self.genes['activation_type_a']=='relu'):
            new_model_modules.append(nn.ReLU())
        elif(self.genes['activation_type_a']=='elu'):
            new_model_modules.append(nn.ELU())
        elif(self.genes['activation_type_a']=='selu'):
            new_model_modules.append(nn.SELU())
        else:
            new_model_modules.append(nn.Tanh())
        if(self.genes['include_pool_a'] and not self.genes['skip_connection']):
            if(self.out_dimensions<2):
                self.fitness = 0
                return nn.Sequential()
            if(self.genes['pool_type_a']=='max_pooling'):
                new_model_modules.append(nn.MaxPool2d(2,2,padding = padding_size))
                # new_model_modules.append(nn.MaxPool2d(2,2,padding = 'same'))
                self.out_dimensions = self.out_dimensions//2
            elif(self.genes['pool_type_a']=='avg_pooling'):
                new_model_modules.append(nn.AvgPool2d(2,2,padding = padding_size))
                # new_model_modules.append(nn.AvgPool2d(2,2,padding = 'same'))
                self.out_dimensions = self.out_dimensions//2
            else:
                raise Exception('Invalid pool type (a layer)')
        
        if(self.genes['include_BN_a']):
            new_model_modules.append(nn.BatchNorm2d(self.genes['out_channels_a']))
        
        if(self.genes['include_b']):
            if(self.out_dimensions<self.genes['k_size_b']):
                self.fitness = 0
                return nn.Sequential()
            # layer_b = nn.Conv2d(self.genes['out_channels_a'],self.genes['out_channels_b'],self.genes['k_size_b'],padding = self.genes['k_size_b']//2 if self.genes['skip_connection'] else 0)
            layer_b = nn.Conv2d(self.genes['out_channels_a'],self.genes['out_channels_b'],self.genes['k_size_b'],padding = 'same')
            # self.out_dimensions = (self.out_dimensions-self.genes['k_size_b']+1)
            new_model_modules.append(layer_b)
            if(self.genes['activation_type_b']=='relu'):
                new_model_modules.append(nn.ReLU())
            elif(self.genes['activation_type_b']=='elu'):
                new_model_modules.append(nn.ELU())
            elif(self.genes['activation_type_b']=='selu'):
                new_model_modules.append(nn.SELU())
            else:
                new_model_modules.append(nn.Tanh())
            
            if(self.genes['include_pool_b'] and not self.genes['skip_connection']):
                if(self.out_dimensions<2):
                    self.fitness = 0
                    return nn.Sequential()
                if(self.genes['pool_type_b']=='max_pooling'):
                    new_model_modules.append(nn.MaxPool2d(2,2,padding = padding_size))
                    # new_model_modules.append(nn.MaxPool2d(2,2,padding = 'same'))
                    self.out_dimensions = self.out_dimensions//2
                elif(self.genes['pool_type_b']=='avg_pooling'):
                    new_model_modules.append(nn.AvgPool2d(2,2,padding = padding_size))
                    # new_model_modules.append(nn.AvgPool2d(2,2,padding = 'same'))
                    self.out_dimensions = self.out_dimensions//2
                else:
                    raise Exception('Invalid pool type (b layer)')
                
            if(self.genes['include_BN_b']):
                new_model_modules.append(nn.BatchNorm2d(self.genes['out_channels_b']))
                
        if(self.genes['include_c']):
            if(self.out_dimensions<self.genes['k_size_c']):
                self.fitness = 0
                return nn.Sequential()
            # layer_b = nn.Conv2d(self.genes['out_channels_a'],self.genes['out_channels_b'],self.genes['k_size_b'],padding = self.genes['k_size_b']//2 if self.genes['skip_connection'] else 0)
            layer_c = nn.Conv2d(self.genes['out_channels_b'],self.genes['out_channels_c'],self.genes['k_size_c'],padding = 'same')
            # self.out_dimensions = (self.out_dimensions-self.genes['k_size_b']+1)
            new_model_modules.append(layer_c)
            if(self.genes['activation_type_c']=='relu'):
                new_model_modules.append(nn.ReLU())
            elif(self.genes['activation_type_c']=='elu'):
                new_model_modules.append(nn.ELU())
            elif(self.genes['activation_type_c']=='selu'):
                new_model_modules.append(nn.SELU())
            else:
                new_model_modules.append(nn.Tanh())
            
            if(self.genes['include_pool_c'] and not self.genes['skip_connection']):
                if(self.out_dimensions<2):
                    self.fitness = 0
                    return nn.Sequential()
                if(self.genes['pool_type_c']=='max_pooling'):
                    new_model_modules.append(nn.MaxPool2d(2,2,padding = padding_size))
                    # new_model_modules.append(nn.MaxPool2d(2,2,padding = 'same'))
                    self.out_dimensions = self.out_dimensions//2
                elif(self.genes['pool_type_c']=='avg_pooling'):
                    new_model_modules.append(nn.AvgPool2d(2,2,padding = padding_size))
                    # new_model_modules.append(nn.AvgPool2d(2,2,padding = 'same'))
                    self.out_dimensions = self.out_dimensions//2
                else:
                    raise Exception('Invalid pool type (c layer)')
                
            if(self.genes['include_BN_c']):
                new_model_modules.append(nn.BatchNorm2d(self.genes['out_channels_c']))
        
        if(self.genes['include_d']):
            if(self.out_dimensions<self.genes['k_size_d']):
                self.fitness = 0
                return nn.Sequential()
            # layer_b = nn.Conv2d(self.genes['out_channels_a'],self.genes['out_channels_b'],self.genes['k_size_b'],padding = self.genes['k_size_b']//2 if self.genes['skip_connection'] else 0)
            layer_d = nn.Conv2d(self.genes['out_channels_c'],self.genes['out_channels_d'],self.genes['k_size_d'],padding = 'same')
            # self.out_dimensions = (self.out_dimensions-self.genes['k_size_b']+1)
            new_model_modules.append(layer_d)
            if(self.genes['activation_type_d']=='relu'):
                new_model_modules.append(nn.ReLU())
            elif(self.genes['activation_type_d']=='elu'):
                new_model_modules.append(nn.ELU())
            elif(self.genes['activation_type_d']=='selu'):
                new_model_modules.append(nn.SELU())
            else:
                new_model_modules.append(nn.Tanh())
            
            if(self.genes['include_pool_d'] and not self.genes['skip_connection']):
                if(self.out_dimensions<2):
                    self.fitness = 0
                    return nn.Sequential()
                if(self.genes['pool_type_d']=='max_pooling'):
                    new_model_modules.append(nn.MaxPool2d(2,2,padding = padding_size))
                    # new_model_modules.append(nn.MaxPool2d(2,2,padding = 'same'))
                    self.out_dimensions = self.out_dimensions//2
                elif(self.genes['pool_type_d']=='avg_pooling'):
                    new_model_modules.append(nn.AvgPool2d(2,2,padding = padding_size))
                    # new_model_modules.append(nn.AvgPool2d(2,2,padding = 'same'))
                    self.out_dimensions = self.out_dimensions//2
                else:
                    raise Exception('Invalid pool type (d layer)')
                
            if(self.genes['include_BN_d']):
                new_model_modules.append(nn.BatchNorm2d(self.genes['out_channels_d']))
        if(self.phase!=0):
            new_model = nn.Sequential(prev_best_model,*new_model_modules)
        else:
            new_model = nn.Sequential(*new_model_modules)
        if(self.genes['skip_connection']):
            self.out_dimensions = 32 if self.phase==0 else self.prev_best.out_dimensions
        # print(new_model)
        return new_model            
 
    def fitness_function(self,train_loader,test_loader)->float:
        
        new_model = FinalModel(self)
        #Training loop
        optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        criterion = F.nll_loss
        new_model.to(self.device)
        num_epochs = 1
        for epoch in range(num_epochs):
            pbar = tqdm(train_loader)
            new_model.train()
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = new_model(x = data, chromosome = self)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                pbar.set_description(desc= f'epoch {epoch} loss={loss.item()} batch_id={batch_idx}')
            # Training accuracy
            '''
            correct = 0
            total = 0
            new_model.eval()
            with torch.no_grad():
                for data in train_loader:
                    images, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = new_model(images,self)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print("Training accuracy: {}".format(100 * correct / total))
            '''
            #Testing loop
            correct = 0
            total = 0
            new_model.eval()
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = new_model(images,self)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print("Validation accuracy: {}".format(100 * correct / total))
        print(f"Fitness calculated: {100 * correct / total}")
        return 100 * correct / total
 
    def crossover(self, chromosome):
        genes1 = self.genes
        genes2 = chromosome.genes
        keys = genes1.keys()
        new_genes = {}
        for key in keys:
            new_genes[key] = random.choice([genes1[key], genes2[key]])
        new_chromosome = Chromosome(self.phase, self.prev_best, new_genes, self.train_loader, self.test_loader)
        return new_chromosome 
    
    def mutation(self):
        mutated_gene = random.choice(list(self.genes.keys()))
        possible_values = [value for value in SEARCH_SPACE[mutated_gene]]
        possible_values.remove(self.genes[mutated_gene])
        new_gene_value = random.choice(possible_values)
        new_genes = self.genes.copy()
        new_genes[mutated_gene] = new_gene_value
        new_chromosome = Chromosome(self.phase, self.prev_best, new_genes, self.train_loader, self.test_loader)
        return new_chromosome
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
        num_layers = random.randrange(1,4)
        if num_layers == 4:
            gene['include_a'] = True
            gene['include_b'] = True
            gene['include_c'] = True
            gene['include_d'] = True
        elif num_layers == 3:
            gene['include_a'] = True
            gene['include_b'] = True
            gene['include_c'] = True
            gene['include_d'] = False
        elif num_layers == 2:
            gene['include_a'] = True
            gene['include_b'] = True
            gene['include_c'] = False
            gene['include_d'] = False
        else:
            gene['include_a'] = True
            gene['include_b'] = False
            gene['include_c'] = False
            gene['include_d'] = False
        return gene
 
    def sort_pop(self):
        sorted_pop = sorted(self.pop,
                            key=lambda x: x.fitness,
                            reverse=True)
        self.pop = sorted_pop
 
    def generate(self):
        # print("start gen")
        self.sort_pop()
        # print(f"{[i.fitness for i in self.pop]}")
        num_fit_selected = int(self.fit_survival_rate * self.pop_size)
        num_unfit_selected = int(self.unfit_survival_rate * self.pop_size)
        num_mutate = int(self.mutation_rate * self.pop_size)
 
        new_pop = []
 
        for i in range(num_fit_selected):
            if(self.pop[i].fitness!=0):
                new_pop.append(self.pop[i])
 
        # print('ok')
 
 
        for i in range(num_unfit_selected):
            # print(i)
            if(self.pop[self.pop_size-i-1].fitness!=0):
                new_pop.append(self.pop[self.pop_size - i - 1])
 
        if (num_mutate > len(new_pop)):
            indices_to_mutate = random.sample(
                range(0, len(new_pop)), len(new_pop))
        else:
            indices_to_mutate = random.sample(
                range(0, len(new_pop)), num_mutate)
        
        for i in indices_to_mutate:
            if(new_pop[i].fitness!=0):
                new_pop[i] = new_pop[i].mutation()
 
        # print("Mutuation done.", [i.fitness for i in new_pop])
 
        parents_list = []
        for i in range(self.pop_size - len(new_pop)):
            parents = random.sample(range(0, len(new_pop)), 2)
            parents_list.append(tuple(parents))
 
        for p1, p2 in parents_list:
            if(new_pop[p1].fitness!=0 and new_pop[p2].fitness!=0):
                new_pop.append(new_pop[p1].crossover(new_pop[p2]))
 
        self.pop = new_pop
        self.pop_size = len(new_pop)
        self.sort_pop()
        # print(self.pop_size)
        print("\n\n")
        # print(f"{[i.fitness for i in self.pop]}")
 
    def find_fittest(self):
        self.sort_pop()
        return self.pop[0]
num_individuals = 15
generation = Generation(fit_survival_rate = FIT_SURVIVAL_RATE,
                        unfit_survival_rate = UNFIT_SURVIVAL_RATE,
                        mutation_rate = MUTATION_RATE,
                        pop_size = num_individuals,
                        phase = 0,
                        search_space = SEARCH_SPACE,
                        prev_best = None,
                        train_loader = train_loader,
                        test_loader = test_loader)

#now, from the generation of individuals created, we seek to perform a tournament in which two individuals are taken -- the less fit one is killed, while the more fit one is mutated.

rounds = 15

for i in range (rounds):
   index1 = random.randrange(0, generation.pop_size)
   flag = True
   while(flag):
     index2 = random.randrange(0, generation.pop_size)
     if(index2 > index1):
         flag = False
    
   if(generation.pop[index1].fitness > generation.pop[index2].fitness):

       generation.pop.append(generation.pop[index1].mutation())
       #Kill the index2
       generation.pop.pop(index2)
       #Reproduce the first one

   else:
        generation.pop.append(generation.pop[index2].mutation())
        #Kill the index1
        generation.pop.pop(index1)
        #Reproduce the second one

generation.sort_pop()
print("fittest is; ", generation.find_fittest().fitness)

   

    
