make && ./main -algorithm DistMult -model_path output-distmult-15.model
./main -algorithm DistMult -model_path output-distmult-15.model -prediction 1

make && ./main -algorithm Analogy -model_path output-analogy-15.model
./main -algorithm Analogy -model_path output-analogy-15.model -prediction 1

make && ./main -algorithm Complex -model_path output-complex-15.model
./main -algorithm Complex -model_path output-complex-15.model -prediction 1
