package NeuralNet.Costs;

import org.la4j.Matrix;
import org.la4j.vector.functor.VectorFunction;


public abstract class Loss {
	public VectorFunction loss;
	public VectorFunction derivative;
	
	public double computeLoss(Matrix prediction, Matrix truth) {
		throw new IllegalArgumentException("Should not call this abstract method!");
	}

	public Matrix computeGradient(Matrix prediction, Matrix truth) {
		throw new IllegalArgumentException("Should not call this abstract method!");
	}
}
