cd ..
python training.py -op one_class -pt 0 -la 4 -ra 0.1 -lb 2 -gpu 2
python training.py -op one_class -pt 0 -la 4 -ra 0.5 -lb 2 -gpu 2
python training.py -op one_class -pt 0 -la 4 -ra 1.0 -lb 2 -gpu 2

python training.py -op one_class -pt 1 -la 4 -ra 0.1 -lb 2 -gpu 2
python training.py -op one_class -pt 1 -la 4 -ra 0.5 -lb 2 -gpu 2
python training.py -op one_class -pt 1 -la 4 -ra 1.0 -lb 2 -gpu 2

python training.py -op one_class -pt 0 -la 7 -ra 0.1 -lb 2 -gpu 2
python training.py -op one_class -pt 0 -la 7 -ra 0.5 -lb 2 -gpu 2
python training.py -op one_class -pt 0 -la 7 -ra 1.0 -lb 2 -gpu 2

python training.py -op one_class -pt 1 -la 7 -ra 0.1 -lb 2 -gpu 2
python training.py -op one_class -pt 1 -la 7 -ra 0.5 -lb 2 -gpu 2
python training.py -op one_class -pt 1 -la 7 -ra 1.0 -lb 2 -gpu 2
