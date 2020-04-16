cd ..
python training.py -op rec -lb 0 -la 6 -ra 0.1 -nt fmnist_LeNet_rec -gpu 2
python training.py -op rec -lb 0 -la 6 -ra 0.5 -nt fmnist_LeNet_rec -gpu 2
python training.py -op rec -lb 0 -la 6 -ra 1.0 -nt fmnist_LeNet_rec -gpu 2

python training.py -op rec -lb 0 -la 7 -ra 0.1 -nt fmnist_LeNet_rec -gpu 2
python training.py -op rec -lb 0 -la 7 -ra 0.5 -nt fmnist_LeNet_rec -gpu 2
python training.py -op rec -lb 0 -la 7 -ra 1.0 -nt fmnist_LeNet_rec -gpu 2
