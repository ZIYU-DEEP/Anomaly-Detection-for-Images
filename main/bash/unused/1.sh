cd ..
python training.py -op rec -lb 4 -la 2 -ra 0.1 -nt fmnist_LeNet_rec -gpu 0
python training.py -op rec -lb 4 -la 2 -ra 0.5 -nt fmnist_LeNet_rec -gpu 0
python training.py -op rec -lb 4 -la 2 -ra 1.0 -nt fmnist_LeNet_rec -gpu 0

python training.py -op rec -lb 4 -la 5 -ra 0.1 -nt fmnist_LeNet_rec -gpu 0
python training.py -op rec -lb 4 -la 5 -ra 0.5 -nt fmnist_LeNet_rec -gpu 0
python training.py -op rec -lb 4 -la 5 -ra 1.0 -nt fmnist_LeNet_rec -gpu 0
