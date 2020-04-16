cd ..
python training.py -op rec -lb 2 -la 4 -ra 0.1 -nt fmnist_LeNet_rec -gpu 1
python training.py -op rec -lb 2 -la 4 -ra 0.5 -nt fmnist_LeNet_rec -gpu 1
python training.py -op rec -lb 2 -la 4 -ra 1.0 -nt fmnist_LeNet_rec -gpu 1

python training.py -op rec -lb 2 -la 7 -ra 0.1 -nt fmnist_LeNet_rec -gpu 1
python training.py -op rec -lb 2 -la 7 -ra 0.5 -nt fmnist_LeNet_rec -gpu 1
python training.py -op rec -lb 2 -la 7 -ra 1.0 -nt fmnist_LeNet_rec -gpu 1
