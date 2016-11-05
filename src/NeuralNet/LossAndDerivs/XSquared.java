package NeuralNet.LossAndDerivs;

import org.la4j.matrix.functor.MatrixFunction;

public class XSquared implements MatrixFunction {

	@Override
	public double evaluate(int i, int j, double value) {

		return value*value;
	}

}