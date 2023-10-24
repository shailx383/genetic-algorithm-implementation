import numpy
import torch.nn as nn
class Chromosome:
    def __init__(self, phase, genes, prev_best = None):
        self.phase = phase
        self.prev_best: Chromosome = prev_best
        self.genes = genes
        # genes["k_size_a"] = k_size_a
        # genes["k_size_b"] = k_size_b
        # genes["include_b"] = include_b
        # genes["include_BN_a"] = include_BN_a
        # genes["include_BN_b"] = include_BN_b
        # genes["out_channels_a"] = out_channels_a
        # genes["out_channels_b"] = out_channels_b
        # genes["include_pool_a"] = include_pool_a
        # genes["include_pool_b"] = include_pool_b
        # genes["pooltype_a"] = pooltype_a
        # genes["pooltype_b"] = pooltype_b
        # genes["skip"] = skip
        self.genes = genes
        if(prev_best == None):
            self.out_dimensions = 32
        else:
            self.out_dimensions = self.prev_best.out_dimension
        self.model = self.build_model()
    def build_model(self):
        layers = []
        out_channels_a = self.genes['out_channels_a']
        k_size_a = self.genes['k_size_a']
        include_BN_a = self.genes['include_BN_a']
        in_channels_a = 3
        if(self.phase > 0):
            in_channels_a = self.prev_best.genes['out_channels_a']
            if(self.prev_best.genes['include_b'] == True):
                in_channels_a = self.prev_best.genes['out_channels_b']
            out_channels_a = self.genes['out_channels_a']
        else:
            in_channels_a = 3
        include_pool_a = self.genes['include_pool_a']
        pool_type_a = self.genes['pool_type_a']
        layer_a = nn.Conv2d(in_channels_a, out_channels_a, k_size_a)
        layers.append(layer_a)
        self.out_dimensions = self.out_dimensions - k_size_a + 1
        activation_type_a = self.genes['activation_type_a']
        activation_a = nn.ReLU()
        if(activation_type_a == 'relu'):
            activation_a = nn.ReLU()
        # elif(activation_type_a == '')
        layers.append(activation_a)
        if(include_pool_a == True):
            if(pool_type_a == 'max'):
                pool_layer_a = nn.MaxPool2d(2,2)
                self.out_dimensions/=2
            layers.append(pool_layer_a)
        if(include_BN_a == True):
                BN_a = nn.BatchNorm2d(out_channels_a)
                layers.append(BN_a)
        if(self.genes['include_b'] == True):
            k_size_b = self.genes['k_size_b']
            include_BN_b = self.genes['include_BN_b']
            in_channels_b = out_channels_a
            out_channels_b = self.genes['out_channels_b']
            include_pool_b = self.genes['include_pool_b']
            pool_type_b = self.genes['pool_type_b']
            layer_b = nn.Conv2d(in_channels_b,out_channels_b, k_size_b)
            layers.append(layer_b)
            self.out_dimensions = self.out_dimensions - k_size_b + 1
            activation_type_b = self.genes['activation_type_a']
            activation_b = nn.ReLU()
            if(activation_type_b == 'relu'):
                activation_b = nn.ReLU()
                layers.append(activation_b)
            if(include_pool_b == True):
                if(pool_type_b == 'max'):
                    pool_layer_b = nn.MaxPool2d(2,2)
                self.out_dimensions/=2
                layers.append(pool_layer_b)
            if(include_BN_b == True):
                BN_b = nn.BatchNorm2d(out_channels_b)
                layers.append(BN_b)
        return nn.Sequential(*layers)