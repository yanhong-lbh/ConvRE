# Replicating Results for "Convolutional Knowledge Graph Link Prediction with Reshaped Embeddings"

This guide will explain how to replicate the results obtained in the paper "Convolutional Knowledge Graph Link Prediction with Reshaped Embeddings". 

## Installation
Follow the below steps to install all the necessary components.

**Step 1: Install PyTorch using Anaconda**
Install Anaconda from this [link](https://www.anaconda.com/products/distribution). Once Anaconda is installed, you can install PyTorch by using the following command:
```
conda install pytorch torchvision torchaudio -c pytorch
```

**Step 2: Clone the repository**
You can clone the repository by using the following command:
```
git clone https://github.com/TimDettmers/ConvE.git
cd ConvE
```

**Step 3: Install the requirements**
You can install all the requirements by using the following command:
```
pip install -r requirements.txt
```

**Step 4: Download the default English model**
To download the English model, run the following commands:
```
python -m spacy download en_core_web_sm
pip install en_core_web_sm-3.2.0.tar.gz
pip install spacy==3.2.0
```

**Step 5: Run the preprocessing script**
You can run the preprocessing script by using the following command:
```
cd ConvE
sh preprocess.sh
```

## Running Experiments
Now you can run experiments with the following commands:

**Experiment with FB15k-237:**
The embedding shape (N, M) is shown before each command, N represents the number of rows and M represents the number of columns in the reshaped embedding.

*(50, 4)*
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data FB15k-237 --input-drop 0.2 --hidden-drop 0.5 --feat-drop 0.2 --lr 0.003 --preprocess --embedding-shape1 50 --hidden-size 6272 --epochs 502
```
*(40, 5)*
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data FB15k-237 --input-drop 0.2 --hidden-drop 0.5 --feat-drop 0.2 --lr 0.003 --preprocess --embedding-shape1 40 --hidden-size 7488 --epochs 502
```
*(25, 8)*
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data FB15k-237 --input-drop 0.2 --hidden-drop 0.5 --feat-drop 0.2 --lr 0.003 --preprocess --embedding-shape1 25 --hidden-size 9216 --epochs 502
```
*(20, 10)*
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data FB15k-237 --input-drop 0.2 --hidden-drop 0.5 --feat-drop 0.2 --lr 0.003 --preprocess --embedding-shape1 20 --epochs 502
```
*(10, 20)*
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data FB15k-237 --input-drop 0.2 --hidden-drop 0.5 --feat-drop 0.2 --lr 0.003 --preprocess --embedding-shape1 10 --hidden-size 10368 --epochs 502
```
*(8, 25)*
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve

 --data FB15k-237 --input-drop 0.2 --hidden-drop 0.5 --feat-drop 0.2 --lr 0.003 --preprocess --embedding-shape1 8 --hidden-size 10304 --epochs 502
```
*(5, 40)*
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data FB15k-237 --input-drop 0.2 --hidden-drop 0.5 --feat-drop 0.2 --lr 0.003 --preprocess --embedding-shape1 5 --epochs 502
```
*(4, 50)*
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data FB15k-237 --input-drop 0.2 --hidden-drop 0.5 --feat-drop 0.2 --lr 0.003 --preprocess --embedding-shape1 4 --hidden-size 9216 --epochs 502 
```
*(2, 100)*
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data FB15k-237 --input-drop 0.2 --hidden-drop 0.5 --feat-drop 0.2 --lr 0.003 --preprocess --embedding-shape1 2 --hidden-size 6272 --epochs 500
```

**Experiment with WN18RR:**

*(50, 4)*
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data WN18RR --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2 --lr 0.003 --preprocess --embedding-shape1 50 --hidden-size 6272 --preprocess 
```
*(40, 5)*
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data WN18RR --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2 --lr 0.003 --preprocess --embedding-shape1 40 --hidden-size 7488
```
*(25, 8)*
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data WN18RR --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2 --lr 0.003 --preprocess --embedding-shape1 25 --hidden-size 9216
```
*(20, 10)*
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data WN18RR --input-drop 0.2 --hidden-drop 0.5 --feat-drop 0.2 --lr 0.003 --preprocess --embedding-shape1 20
```
*(10, 20)*
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data WN18RR --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2 --lr 0.003 --preprocess --embedding-shape1 10 --hidden-size 10368
```
*(8, 25)*
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data WN18RR --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2 --lr 0.003 --preprocess --embedding-shape1 8 --hidden-size 10304
```
*(5, 40)*
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data WN18RR --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.

2 --lr 0.003 --preprocess --embedding-shape1 5
```
*(4, 50)*
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data WN18RR --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2 --lr 0.003 --preprocess --embedding-shape1 4 --hidden-size 9216
```
*(2, 100)*
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data WN18RR --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2 --lr 0.003 --preprocess --embedding-shape1 2 --hidden-size 6272
```

## Credits

This work is based on the code from the [ConvE repository](https://github.com/TimDettmers/ConvE) and the tutorial at this [CSDN blog post](https://blog.csdn.net/qq_40506723/article/details/123315377).
