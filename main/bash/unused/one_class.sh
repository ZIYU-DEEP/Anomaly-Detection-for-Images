cd ..
python training.py -pt 0 -ra 0.
python training.py -pt 1 -ra 0.

python training.py -op one_class -pt 0 -la 1 -ra 0.1
python training.py -op one_class -pt 0 -la 1 -ra 0.5
python training.py -op one_class -pt 0 -la 1 -ra 1.0

python training.py -op one_class -pt 1 -la 1 -ra 0.1
python training.py -op one_class -pt 1 -la 1 -ra 0.5
python training.py -op one_class -pt 1 -la 1 -ra 1.0
