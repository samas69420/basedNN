# basedNN

### Yet another collection of neural networks implementations from scratch

built with nothing but grit, tears, and raw Python \
(using python lists as tensors 💀)

\
![Python Version](https://img.shields.io/badge/Python-3.13%2B-blue)  
![Efficiency Badge](https://img.shields.io/badge/Efficiency-LOL%20NOPE-red)  

---

## Why?  
*"Why build neural networks from scratch in 2025?"*

Because libraries are for the weak 🗿 

but also because i wanted a repo with simple and easily readable code to check when i need to refresh the implementation details of the backprop algorithm

---

## Features
- ✅ **No external libraries**: not even NumPy, we do matrix math like it’s 1998  
- ✅ **Artisanal backpropagation**: 100% gluten free handcrafted gradients  
- ✅ **Readability**: so simple, even your toaster could understand it  
- ❌ **Efficiency**: training MNIST? *Estimated completion: 2047* 

---

## Usage
to start training an architecture cd into its directory and from there run the main.py in src folder
for example to test the mlp:
> ```bash  
>$ git clone https://github.com/samas69420/basedNN
>$ cd basedNN/mlp
>$ python src/main.py   
> ```

the hyperparameters like learning rate, network layers etc can be modified in ```main.py```

---

## Contributing  
1. Fork the repo.  
2. Add more inefficient (but easy to read) code.  
3. ???.  
4. Profit (not really).  

---

## Ze math
![](mlp/backprop.jpg?raw=true)
![](lstm/bpttlstm.jpg?raw=true)
![](cnn/bpconv.jpg?raw=true)

###### for more information about the notation: https://robotchinwag.com/posts/the-tensor-calculus-you-need-for-deep-learning/

---

### This repo is intended for educational purposes only and shouldn't be used for real world applications, why would it be a bad idea?
- 🚨 **No vectorization**: we loop like it’s a cardio workout  
- 🚨 **No GPU support**: your CPU will hate you  
- 🚨 **No fancy CPU optimizations**: this code is not even multithreaded, with large models or datasets it would be slower than continental drift   

