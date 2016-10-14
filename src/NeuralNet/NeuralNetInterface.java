package NeuralNet;

import org.la4j.Matrix;
import org.la4j.Vector;

public interface NeuralNetInterface {

	public void forwardProp(Matrix batchData);
	public void backwardProp();
	public double runOnePass(Matrix data, Vector truth);
	public void computeGradients(Matrix data);
	
}
