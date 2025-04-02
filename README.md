# basedNN

### Yet another fully-connected feedforward neural network implementation from scratch

built with nothing but grit, tears, and raw Python \
(using python lists as tensors 💀)

\
![Python Version](https://img.shields.io/badge/Python-3.13%2B-blue)  
![Efficiency Badge](https://img.shields.io/badge/Efficiency-LOL%20NOPE-red)  

---

## Why?  
*"Why build a neural network from scratch in 2025?"*

Because libraries are for the weak 🗿 

but also because i wanted a repo with simple and easily readable code to check when i need to refresh the implementation details of the backprop algorithm

---

## Features
- ✅ **Custom activation function**: just define your favorite f:R -> R (and its derivative) in main.py and pass them to the net constructor as arguments
- ✅ **Custom MLP architecture**: change the NETWORK\_LAYERS list in main.py specifying how many layers and how many neurons per layer should be used
- ✅ **No external libraries**: not even NumPy, we do matrix math like it’s 1998  
- ✅ **Artisanal backpropagation**: 100% gluten free handcrafted gradients  
- ✅ **Readability**: so simple, even your toaster could understand it  
- ❌ **Efficiency**: training MNIST? *Estimated completion: 2047* 

---

## Usage
the training could be started with:
> ```bash  
> cd src
> python main.py   
> ```

the hyperparameters like learning rate and network architecture can be modified in ```main.py```

---

## Contributing  
1. Fork the repo.  
2. Add more inefficient (but easy to read) code.  
3. ???.  
4. Profit (not really).  

---

## Ze math
![](backprop.jpg?raw=true)

###### for more information about the notation: https://robotchinwag.com/posts/the-tensor-calculus-you-need-for-deep-learning/

---

### This repo is intended for educational purposes only and shouldn't be used for real world applications, why would it be a bad idea?
- 🚨 **No vectorization**: we loop like it’s a cardio workout  
- 🚨 **No GPU support**: your CPU will hate you  
- 🚨 **No fancy CPU optimizations**: this code is not even multithreaded, with large models or datasets it would be slower than continental drift   

