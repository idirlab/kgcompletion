make && ./main -algorithm DistMult -model_path output-distmult-237.model
./main -algorithm DistMult -model_path output-distmult-237.model -prediction 1

make && ./main -algorithm Analogy -model_path output-analogy-237.model
./main -algorithm Analogy -model_path output-analogy-237.model -prediction 1

make && ./main -algorithm Complex -model_path output-complex-237.model
./main -algorithm Complex -model_path output-complex-237.model -prediction 1
