import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import torch
class Chromosome:
    def __init__(self,phase:int,prev_best,genes:dict):
        self.phase = phase
        self.prev_best = prev_best
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.genes = genes
        self.out_dimensions = prev_best.out_dimensions if phase!=0 else 32
        self.model:nn.Module = self.build_model()
        # self.fitness = self.fitness_function()

    def build_model(self)->nn.Module:
        if(self.prev_best!=None):
            prev_best_model:nn.Module = self.prev_best.model
        new_model_modules = []
        if(self.phase!=0):
            layer_a = nn.Conv2d(self.prev_best.genes['out_channels_b'] if self.prev_best.genes['include_b'] else self.prev_best.genes['out_channels_a'],self.genes['out_channels_a'],self.genes['k_size_a'])
        else:
            layer_a = nn.Conv2d(3,self.genes['out_channels_a'],self.genes['k_size_a'])
        self.out_dimensions = (self.out_dimensions-self.genes['k_size_a']+1)
        new_model_modules.append(layer_a)
        if(self.genes['activation_type_a']=='relu'):
            new_model_modules.append(nn.ReLU())
        else:
            new_model_modules.append(nn.Tanh())
        if(self.genes['include_pool_a']):
            if(self.genes['pool_type_a']=='max_pooling'):
                new_model_modules.append(nn.MaxPool2d(2,2))
                self.out_dimensions = self.out_dimensions//2
            elif(self.genes['pool_type_a']=='avg_pooling'):
                new_model_modules.append(nn.AvgPool2d(2,2))
                self.out_dimensions = self.out_dimensions//2
            else:
                raise Exception('Invalid pool type (a layer)')
        
        if(self.genes['include_BN_a']):
            new_model_modules.append(nn.BatchNorm2d(self.genes['out_channels_a']))
        
        if(self.genes['include_b'] or self.phase==0):
            layer_b = nn.Conv2d(self.genes['out_channels_a'],self.genes['out_channels_b'],self.genes['k_size_b'])
            self.out_dimensions = (self.out_dimensions-self.genes['k_size_b']+1)
            new_model_modules.append(layer_b)
            if(self.genes['activation_type_b']=='relu'):
                new_model_modules.append(nn.ReLU())
            else:
                new_model_modules.append(nn.Tanh())
            
            if(self.genes['include_pool_b']):
                if(self.genes['pool_type_b']=='max_pooling'):
                    new_model_modules.append(nn.MaxPool2d(2,2))
                    self.out_dimensions = self.out_dimensions//2
                elif(self.genes['pool_type_b']=='avg_pooling'):
                    new_model_modules.append(nn.AvgPool2d(2,2))
                    self.out_dimensions = self.out_dimensions//2
                else:
                    raise Exception('Invalid pool type (b layer)')
                
            if(self.genes['include_BN_b']):
                new_model_modules.append(nn.BatchNorm2d(self.genes['out_channels_b']))
        if(self.phase!=0):
            new_model = nn.Sequential(prev_best_model,*new_model_modules)
        else:
            new_model = nn.Sequential(*new_model_modules)
        return new_model            

    def fitness_function(self,train_loader,test_loader)->float:
        self.model = self.model.to(self.device)
        num_epochs = 5
        criterion = F.nll_loss
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        for epoch in range(num_epochs):
            pbar = tqdm(train_loader)      
            self.model.train()
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.forward(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                pbar.set_description(desc= f'epoch: {epoch} loss={loss.item()} batch_id={batch_idx}')

        

        
    def forward(self,x:torch.Tensor):
        if(x.shape[0]==1):
            return self._forward_one_example(x)
        else:
            outputs = torch.zeros(x.shape[0],10,device = self.device)
            for i in range(x.shape[0]):
                outputs[i] = self._forward_one_example(x[i].unsqueeze(0))
            return outputs

    def _forward_one_example(self,x:torch.Tensor):
        x = self.model(x)
        if(self.genes['skip_connection']):
            if(self.phase==0):
                y = nn.Conv2d(3,self.genes['out_channels_b'],1)(x)
            else:
                y = nn.Conv2d(self.prev_best.genes['out_channels_b'] if self.prev_best.genes['include_b'] else self.prev_best.genes['out_channels_a'],self.genes['out_channels_b'] if self.genes['include_b'] else self.genes['out_channels_a'],1)(x)
            x = x+y
        x = x.flatten()
        x = nn.Linear(x.shape[0],10,device=self.device)(x)
        return F.log_softmax(x,dim = 0)
