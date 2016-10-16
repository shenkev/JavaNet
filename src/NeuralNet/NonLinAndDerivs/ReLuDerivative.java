package NeuralNet.NonLinAndDerivs;

import org.la4j.matrix.functor.MatrixFunction;

public class ReLuDerivative implements MatrixFunction {

	@Override
	public double evaluate(int i, int j, double value) {
		
		if (value > 0.0) {
			return 1.0;
		} else {
			return 0.0;
		}
	}

}
