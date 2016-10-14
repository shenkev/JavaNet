package NeuralNet.Costs;

import org.la4j.matrix.functor.MatrixFunction;

public abstract class Loss {
	public MatrixFunction loss;
	public MatrixFunction derivative;
}
