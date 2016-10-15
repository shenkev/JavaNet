package NeuralNet.NonLinAndDerivs;

import org.la4j.matrix.functor.MatrixFunction;

import NeuralNet.NonLinFuncs.Tanh;

public class TanhDerivative implements MatrixFunction {

	@Override
	public double evaluate(int i, int j, double value) {

		return 1-Math.pow(Tanh.evaluateTanh(value), 2);
	}

}
