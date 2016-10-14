package NeuralNet.NonLinAndDerivs;

import org.la4j.matrix.functor.MatrixFunction;

import NeuralNet.NonLinFuncs.Sigmoid;

public class SigmoidDerivative implements MatrixFunction{

	@Override
	public double evaluate(int i, int j, double value) {
		double sig = Sigmoid.evaluateSigmoid(value);
		return sig*(1-sig);
	}

}
