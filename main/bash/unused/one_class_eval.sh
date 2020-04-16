cd ..
python evaluation.py -pt 0 -ra 0.
python evaluation.py -pt 1 -ra 0.

python evaluation.py -op one_class -pt 0 -la 1 -ra 0.1
python evaluation.py -op one_class -pt 0 -la 1 -ra 0.5
python evaluation.py -op one_class -pt 0 -la 1 -ra 1.0

python evaluation.py -op one_class -pt 1 -la 1 -ra 0.1
python evaluation.py -op one_class -pt 1 -la 1 -ra 0.5
python evaluation.py -op one_class -pt 1 -la 1 -ra 1.0
