package NeuralNet.NonLinAndDerivs;

import org.la4j.matrix.functor.MatrixFunction;

import NeuralNet.NonLinFuncs.Sigmoid;

public class SigmoidFunction implements MatrixFunction {

	@Override
	public double evaluate(int i, int j, double value) {
		return Sigmoid.evaluateSigmoid(value);
	}


}
