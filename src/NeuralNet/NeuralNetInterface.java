package NeuralNet;

import org.la4j.Matrix;

public interface NeuralNetInterface {

	public void forwardProp(Matrix batchData);
	public void backwardProp(Matrix truth);
	public double runOnePass(Matrix data, Matrix truth);
	public void computeGradients(Matrix data);
	public Matrix predict(Matrix data);
	
}
