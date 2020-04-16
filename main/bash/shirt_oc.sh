cd ..
python training.py -op one_class -pt 0 -la 2 -ra 0.1 -lb 6 -gpu 0
python training.py -op one_class -pt 0 -la 2 -ra 0.5 -lb 6 -gpu 0
python training.py -op one_class -pt 0 -la 2 -ra 1.0 -lb 6 -gpu 0

python training.py -op one_class -pt 1 -la 2 -ra 0.1 -lb 6 -gpu 0
python training.py -op one_class -pt 1 -la 2 -ra 0.5 -lb 6 -gpu 0
python training.py -op one_class -pt 1 -la 2 -ra 1.0 -lb 6 -gpu 0

python training.py -op one_class -pt 0 -la 7 -ra 0.1 -lb 6 -gpu 0
python training.py -op one_class -pt 0 -la 7 -ra 0.5 -lb 6 -gpu 0
python training.py -op one_class -pt 0 -la 7 -ra 1.0 -lb 6 -gpu 0

python training.py -op one_class -pt 1 -la 7 -ra 0.1 -lb 6 -gpu 0
python training.py -op one_class -pt 1 -la 7 -ra 0.5 -lb 6 -gpu 0
python training.py -op one_class -pt 1 -la 7 -ra 1.0 -lb 6 -gpu 0
