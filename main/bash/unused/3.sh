cd ..
python training.py -op one_class -pt 0 -la 5 -ra 0.1 -lb 4 -gpu 2
python training.py -op one_class -pt 0 -la 5 -ra 0.5 -lb 4 -gpu 2
python training.py -op one_class -pt 0 -la 5 -ra 1.0 -lb 4 -gpu 2

python training.py -op one_class -pt 1 -la 5 -ra 0.1 -lb 4 -gpu 2
python training.py -op one_class -pt 1 -la 5 -ra 0.5 -lb 4 -gpu 2
python training.py -op one_class -pt 1 -la 5 -ra 1.0 -lb 4 -gpu 2
