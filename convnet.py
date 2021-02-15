import numpy as np
import pickle
from collections import OrderedDict
from layers import *

class ConvNet:
    def __init__(self, input_shape=(1,50,50), conv_num=1,
                 conv_param=[{'filter_num':30, 'filter_size':7, 'pad':0, 'stride':1, 'pool_size':2}],
                 hidden_size_list=([100]), output_size=2, weight_init_std=0.01):
        self.conv_num = conv_num
        self.input_shape = input_shape
        filter_num = []
        filter_size = []
        filter_pad = []
        filter_stride = []
        pool_size = []
        conv_output_size = []
        pool_output_size = []
        input_size = []
        for i in range(conv_num):
            filter_num.append(conv_param[i]['filter_num'])
            filter_size.append(conv_param[i]['filter_size'])
            filter_pad.append(conv_param[i]['pad'])
            filter_stride.append(conv_param[i]['stride'])
            pool_size.append(conv_param[i]['pool_size'])

        for i in range(conv_num-1):
            if i == 0:
                input_size.append(input_shape[1])
            conv_output_size.append((input_size[i] - filter_size[i] + 2*filter_pad[i]) / filter_stride[i] + 1)
            pool_output_size.append(conv_output_size[i]/pool_size[i])
            input_size.append(pool_output_size[i])
        conv_output_size.append((input_size[-1] - filter_size[-1] + 2*filter_pad[-1]) / filter_stride[-1] + 1)
        pool_output_size.append(int(filter_num[-1] * (conv_output_size[-1]/pool_size[-1]) * (conv_output_size[-1]/pool_size[-1])))
        
        params_num = len(hidden_size_list) + conv_num + 1
        self.params_num = params_num
        
        self.params = {}
        self.params['W1'] = np.random.randn(filter_num[0], input_shape[0],
                                            filter_size[0], filter_size[0]) * weight_init_std
        self.params['b1'] = np.zeros(filter_num[0])
        for idx in range(1, conv_num):
            self.params['W' + str(idx+1)] = np.random.randn(filter_num[idx], filter_num[idx-1],
                                                            filter_size[idx], filter_size[idx]) * weight_init_std
            self.params['b' + str(idx+1)] = np.zeros(filter_num[idx])
        
        self.params['W' + str(conv_num+1)] = np.random.randn(pool_output_size[conv_num-1], hidden_size_list[0])
        self.params['b' + str(conv_num+1)] = np.zeros(hidden_size_list[0])
        for idx in range(1, len(hidden_size_list)):
            self.params['W' + str(conv_num+1+idx)] = np.random.randn(hidden_size_list[idx-1], hidden_size_list[idx]) * weight_init_std
            self.params['b' + str(conv_num+1+idx)] = np.zeros(hidden_size_list[idx])
        self.params['W' + str(params_num)] = np.random.randn(hidden_size_list[-1], output_size) * weight_init_std
        self.params['b' + str(params_num)] = np.zeros(output_size)
        
        
        self.layers = OrderedDict()
        for idx in range(1, conv_num+1):
            self.layers['Conv' + str(idx)] = Convolution(self.params['W'+str(idx)], self.params['b'+str(idx)],
                                                         conv_param[idx-1]['stride'], conv_param[idx-1]['pad'])
            self.layers['Relu' + str(idx)] = Relu()
            self.layers['Pool' + str(idx)] = Pooling(pool_h=pool_size[idx-1], pool_w=pool_size[idx-1], stride=pool_size[idx-1])
        
        for idx in range(1, len(hidden_size_list)+1):
            idx_p = idx + conv_num
            self.layers['Affine' + str(idx)] = Affine(self.params['W'+str(idx_p)], self.params['b'+str(idx_p)])
            self.layers['Relu' + str(idx_p)] = Relu()
        self.layers['Affine' + str(len(hidden_size_list)+1)] = Affine(self.params['W'+str(params_num)], self.params['b'+str(params_num)])
        
        self.last_layer = SoftmaxWithLoss()
        
    def model(self):
        for idx in range(1, self.conv_num+1):
            print("Conv"+str(idx)+": ", end="")
            print(self.params['W'+str(idx)].shape)
        for idx in range(self.conv_num+1, self.params_num+1):
            print("Affine"+str(idx-self.conv_num)+": ", end="")
            print(self.params['W'+str(idx)].shape)
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def gradient(self, x, t):
        self.loss(x, t)
        
        dout = 1
        dout = self.last_layer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}
        for idx in range(1, self.conv_num+1):
            grads['W'+str(idx)] = self.layers['Conv'+str(idx)].dW
            grads['b'+str(idx)] = self.layers['Conv'+str(idx)].db
        for idx in range(self.conv_num+1, self.params_num+1):
            grads['W'+str(idx)] = self.layers['Affine'+str(idx-self.conv_num)].dW
            grads['b'+str(idx)] = self.layers['Affine'+str(idx-self.conv_num)].db
            
        return grads
    
    def save_params(self, file_name="face_params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
        print("params save complete!: {}".format(file_name))
    
    def load_params(self, file_name="face_params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
            
        for idx in range(1, self.conv_num+1):
            self.layers['Conv'+str(idx)].W = self.params['W'+str(idx)]
            self.layers['Conv'+str(idx)].b = self.params['b'+str(idx)]
            
        for idx in range(self.conv_num+1, self.params_num+1):
            idx_p = idx - self.conv_num
            self.layers['Affine'+str(idx_p)].W = self.params['W'+str(idx)]
            self.layers['Affine'+str(idx_p)].b = self.params['b'+str(idx)]
        
        print("params load complete!: {}".format(file_name))