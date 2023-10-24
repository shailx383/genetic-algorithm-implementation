import torch.nn as nn


class Chromosome:
    def __init__(self,phase:int,prev_best,genes:dict):
        self.phase = phase
        self.prev_best = prev_best
        self.genes = genes
        self.out_dimensions = prev_best.out_dimensions if phase!=0 else 32
        self.model = self.build_model()

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

