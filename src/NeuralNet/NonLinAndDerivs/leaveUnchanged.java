package NeuralNet.NonLinAndDerivs;

import org.la4j.matrix.functor.MatrixFunction;

public class leaveUnchanged implements MatrixFunction {

	@Override
	public double evaluate(int i, int j, double value) {
		
		return value;
	}

}
