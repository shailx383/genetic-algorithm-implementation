import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import torch
import random
SEARCH_SPACE = {
    'k_size_a': [1, 3, 5],
    'k_size_b': [1, 3, 5],
    'out_channels_a': [8, 16, 32, 64],
    'out_channels_b': [8, 16, 32, 64],
    'include_pool_a': [True, False],
    'include_pool_b': [True, False],
    'pool_type_a': ['max_pooling','avg_pooling'],
    'pool_type_b': ['max_pooling','avg_pooling'],
    'activation_type_a': ['relu', 'tanh', 'elu', 'selu'],
    'activation_type_b': ['relu', 'tanh', 'elu', 'selu'], 
    'include_b': [True, False],
    'include_BN_a': [True, False],
    'include_BN_b': [True, False],
    'skip_connection': [True, False],
}

class FinalModel(nn.Module):
    def __init__(self, chromosome):
        super().__init__()
        self.block = chromosome.model
        if chromosome.phase == 0:
            in_channels = 3
            out_channels = chromosome.genes['out_channels_b']
        else:
            if(chromosome.prev_best.genes['include_b']):
                in_channels = chromosome.prev_best.genes['out_channels_b']
            else:
                in_channels = chromosome.prev_best.genes['out_channels_a']
            if(chromosome.genes['include_b']):
                out_channels = chromosome.genes['out_channels_b']
            else:
                out_channels = chromosome.genes['out_channels_a']
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_channels*chromosome.out_dimensions**2,10)

    def forward(self, x, chromosome):
        if chromosome.genes['skip_connection']:
            y = x
            if chromosome.phase != 0:
                y = chromosome.prev_best.model(x)
            y = self.skip(y)
        x=self.block(x)
        if chromosome.genes['skip_connection']:
            x = x + y
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
            layer_a = nn.Conv2d(self.prev_best.genes['out_channels_b'] if self.prev_best.genes['include_b'] else self.prev_best.genes['out_channels_a'],self.genes['out_channels_a'],self.genes['k_size_a'],padding = self.genes['k_size_a']//2 if self.genes['skip_connection'] else 0)
        else:
            layer_a = nn.Conv2d(3,self.genes['out_channels_a'],self.genes['k_size_a'],padding = self.genes['k_size_a']//2 if self.genes['skip_connection'] else 0)
        self.out_dimensions = (self.out_dimensions-self.genes['k_size_a']+1)
        new_model_modules.append(layer_a)
        if(self.genes['activation_type_a']=='relu'):
            new_model_modules.append(nn.ReLU())
        else:
            new_model_modules.append(nn.Tanh())
        if(self.genes['include_pool_a'] and not self.genes['skip_connection']):
            if(self.out_dimensions<2):
                self.fitness = 0
                return nn.Sequential()
            if(self.genes['pool_type_a']=='max_pooling'):
                new_model_modules.append(nn.MaxPool2d(2,2,padding = padding_size))
                self.out_dimensions = self.out_dimensions//2
            elif(self.genes['pool_type_a']=='avg_pooling'):
                new_model_modules.append(nn.AvgPool2d(2,2,padding = padding_size))
                self.out_dimensions = self.out_dimensions//2
            else:
                raise Exception('Invalid pool type (a layer)')
        
        if(self.genes['include_BN_a']):
            new_model_modules.append(nn.BatchNorm2d(self.genes['out_channels_a']))
        
        if(self.genes['include_b'] or self.phase==0):
            if(self.out_dimensions<self.genes['k_size_b']):
                self.fitness = 0
                return nn.Sequential()
            layer_b = nn.Conv2d(self.genes['out_channels_a'],self.genes['out_channels_b'],self.genes['k_size_b'],padding = self.genes['k_size_b']//2 if self.genes['skip_connection'] else 0)
            self.out_dimensions = (self.out_dimensions-self.genes['k_size_b']+1)
            new_model_modules.append(layer_b)
            if(self.genes['activation_type_b']=='relu'):
                new_model_modules.append(nn.ReLU())
            else:
                new_model_modules.append(nn.Tanh())
            
            if(self.genes['include_pool_b'] and not self.genes['skip_connection']):
                if(self.out_dimensions<2):
                    self.fitness = 0
                    return nn.Sequential()
                if(self.genes['pool_type_b']=='max_pooling'):
                    new_model_modules.append(nn.MaxPool2d(2,2,padding = padding_size))
                    self.out_dimensions = self.out_dimensions//2
                elif(self.genes['pool_type_b']=='avg_pooling'):
                    new_model_modules.append(nn.AvgPool2d(2,2,padding = padding_size))
                    self.out_dimensions = self.out_dimensions//2
                else:
                    raise Exception('Invalid pool type (b layer)')
                
            if(self.genes['include_BN_b']):
                new_model_modules.append(nn.BatchNorm2d(self.genes['out_channels_b']))
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
        num_epochs = 5
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