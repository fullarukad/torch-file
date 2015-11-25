
require 'image'
require 'nn'

--cargamos los datos
trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')

--un diccionario que contine las clases 
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

--imprmimos para ver la cantidad de datos 
print(trainset)
print(#trainset.data)

--por diversion imprimimos una imagen

--itorch.image(trainset.data[100]) -- display the 100-th image in dataset
--print(classes[trainset.label[100]])

--**no entender esto**--

-- ignore setmetatable for now, it is a feature beyond the scope of this tutorial. It sets the index operator.
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);


trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
    return self.data:size(1) 
end

print(trainset:size()) -- just to test

-----**------------------**-


redChannel = trainset.data[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}
print(#redChannel)

--ahora hallaremos la media y la desviacion estandar para cada canal de las imagen
--para que nuestro dataset quede normalizado



mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end


---ahora crearemos nuestro Neural networks

net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.Linear(120, 84))
net:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems


--definimos nuestra loss function que este caso es ClassNllCriterion

criterion = nn.ClassNLLCriterion()

--el entrenamiento de la red haciendo por la funcion de StochasticGradient

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001 ---ak defino mi learningRate
trainer.maxIteration = 5 -- just do 5 epochs of training.

trainer:train(trainset)


---hasta ak llevamos entrenando nuestra red neuronal con los dataset de entrenamiento
---ahora probaremos la eficiencia de este entrenamiento

--primero visualizamos una imagen desde el testdata
--print(classes[testset.label[100]])
--itorch.image(testset.data[100])
--Nota en mi casa no sirve itorch.image solo sirve en jupyter

--ahora convertiremos los datos del testset en double
--y normalizamos los datos 

testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
for i=1,3 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- for fun, print the mean and standard-deviation of example-100
horse = testset.data[100]
print(horse:mean(), horse:std())



