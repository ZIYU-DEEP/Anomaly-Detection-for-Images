cd ..
python training.py -op one_class -pt 0 -la 6 -ra 0.1 -lb 0 -gpu 3
python training.py -op one_class -pt 0 -la 6 -ra 0.5 -lb 0 -gpu 3
python training.py -op one_class -pt 0 -la 6 -ra 1.0 -lb 0 -gpu 3

python training.py -op one_class -pt 1 -la 6 -ra 0.1 -lb 0 -gpu 3
python training.py -op one_class -pt 1 -la 6 -ra 0.5 -lb 0 -gpu 3
python training.py -op one_class -pt 1 -la 6 -ra 1.0 -lb 0 -gpu 3

python training.py -op one_class -pt 0 -la 7 -ra 0.1 -lb 0 -gpu 3
python training.py -op one_class -pt 0 -la 7 -ra 0.5 -lb 0 -gpu 3
python training.py -op one_class -pt 0 -la 7 -ra 1.0 -lb 0 -gpu 3

python training.py -op one_class -pt 1 -la 7 -ra 0.1 -lb 0 -gpu 3
python training.py -op one_class -pt 1 -la 7 -ra 0.5 -lb 0 -gpu 3
python training.py -op one_class -pt 1 -la 7 -ra 1.0 -lb 0 -gpu 3
