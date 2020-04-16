cd ..
python evaluation.py -ra 0. -nt fmnist_LeNet_rec -op rec_unsupervised -gpu 2

python evaluation.py -op rec -la 1 -ra 0.1 -nt fmnist_LeNet_rec -gpu 2
python evaluation.py -op rec -la 1 -ra 0.5 -nt fmnist_LeNet_rec -gpu 2
python evaluation.py -op rec -la 1 -ra 1.0 -nt fmnist_LeNet_rec -gpu 2
