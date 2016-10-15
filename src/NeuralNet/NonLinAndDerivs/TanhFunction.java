package NeuralNet.NonLinAndDerivs;

import org.la4j.matrix.functor.MatrixFunction;

import NeuralNet.NonLinFuncs.Tanh;

public class TanhFunction implements MatrixFunction {

	@Override
	public double evaluate(int i, int j, double value) {

		return Tanh.evaluateTanh(value);
	}

}
