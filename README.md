# Dinosaur-Land

In this repository, I have coded a character level RNN Recurrent Neural Network *from scratch* to learn the Dinosaur Names
and predict new Dinosaur Names!! <br/>
The RNN coded can be used to save the trained weights as well as load the weights from previous training!!
Exciting!!

## How to use

1. Clone the repository and cd into it
```
git clone https://github.com/iArunava/Dinosaur-Land.git
cd Dinosaur-Land/
```

2. Get the dataset
```
python3 ./dataset/get_dinosaur_names.py
```

3. Train the model (and save [by default] `-s 0` to not save model)
```
python3 init.py -lr 0.03 -e 50000
```

4. Sample from the saved model
```
python3 init.py -t 0
```

5. To set more options use
```
python3 init.py --help
```

## Some of the sampled examples

```
[INFO]Getting New Dinos using saved models
[INFO]Trying to Load the weights with extension "1"
[INFO]Weights Loaded successfully!

Orkstosaurus
Nquiptonsaurus
Gamugaveininnasaurus
Eongboven
Ganstonnosaurus

[INFO]Sampling Complete!! 
 How do you like the new dinosaur names?
[INFO]Exiting...
```

## References

This project is made closely following the deeplearning.ai course from Andrew NG

## LICENSE

The code in this repository is distributed under MIT License. <br/>
Have fun!!
