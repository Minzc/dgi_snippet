This is a demo code of weird memory consumption of dgl
This code is an implement of GCMC to predict scalar rating given user and item.

Dataset is ML-100k with 944 users and 1683 items


dgl=0.62
pytorch=1.60

run:
`python train.py --do_squeeze`


If not squeezing:
`python train.py`
The code will try to allocate 19.31GB Memory.


The do_squeeze argument decides whether the predictor squeeze output shape [batch_size * 1] to [batch_size] at line 199.
