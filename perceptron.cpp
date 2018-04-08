#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <random> 
#include <cmath>
#include <iomanip>


class Neuron{
	private:
		double *weights;
		double threshold;
		double learningRate;
		int correct;
		double activation(std::vector<double> x){
			double sum = 0;
			for(int i = 0; i < 4; i++){
				sum += (x[i] * weights[i]);
			}

			//sign function
			return (sum >= threshold ? 1 : -1);
		}
		void adjustWeights(std::vector<double> x, double yd, double y){
			for(int i = 0; i < 4; i++){
				double dWeight = learningRate * x[i] * (yd-y);
				weights[i] = weights[i] + dWeight;
			}
		}
	public:
		Neuron(){
			weights = new double[4];

		    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    		std::uniform_real_distribution<> dis(-0.5, 0.5); //random number from -0.5 to 0.5

			//initialize weights with values from [-0.5, 0.5]
			weights[0] = roundf(dis(gen) * 10) / 10; 
			weights[1] = roundf(dis(gen) * 10) / 10;
			weights[2] = roundf(dis(gen) * 10) / 10;
			weights[3] = roundf(dis(gen) * 10) / 10;
			threshold = roundf(dis(gen) * 10) / 10;
			learningRate = 0.05f;
			correct = 0;
		}
		void execute(std::vector<double> x, double label){
			//get output label
			double Y = activation(x);

			//check label
			if(Y == label){
				std::cout << "classification was CORRECT!" << std::endl;
				correct++;
			}
			else{
				std::cout << "classification was an ERROR!" << std::endl;
				adjustWeights(x, label, Y);
			}
		}
		~Neuron(){
			delete [] weights;
		}
		void log(){
			std::cout << "Weights are [" 
			<< weights[0] << ", " 
			<< weights[1] << ", " 
			<< weights[2] << ", " 
			<< weights[3] << "], ";
		}
		int getCorrect() const {
			return correct;
		}
};

void loadCSVtoDataset();

std::vector<std::vector<double>> features; // Vector containing features (1-4)
std::vector<double> labels;				   //Vector data holding labels from csv(5)
std::ifstream file;			   // File to load data

int main(int argc, char *argv[]) {
	file.open(argv[1]);
	loadCSVtoDataset();
	Neuron *neuron = new Neuron;
	for (int i = 0; i < features.size(); i++) {
		std::cout << "Iteration " << std::setw(4) << std::left << i+1 << " ";
		neuron->log();
		neuron->execute(features[i], labels[i]);
	}
	std::cout << ((double)neuron->getCorrect()/features.size()*100) << "% Correct!" << std::endl;
	delete neuron;
}

void loadCSVtoDataset(){
	/* 
	* Code to load in from a CSV
	*/
	std::string line;
	while(std::getline(file, line)) {
	    std::vector<double> row;
	    std::stringstream iss(line);

	    std::string val;
	    while (std::getline(iss, val, ',')) {
			try {
		    	double temp = std::stod(val);
		    	row.push_back(temp);
			}
			catch (...) {
		    	std::cout << "Bad input: " << val << std::endl;
			}
	    }
	    features.push_back(row);
	}

	//extract labels
	for(int i = 0; i < features.size(); i++){
		labels.push_back(features[i][features[i].size()-1]);	//add label to labels vector
		features[i].pop_back();
	}
}
