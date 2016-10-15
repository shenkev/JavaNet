package NeuralNet.NonLinFuncs;

import NeuralNet.NonLinAndDerivs.TanhDerivative;
import NeuralNet.NonLinAndDerivs.TanhFunction;

// Outputs -1 to 1
public class Tanh extends NonLinFunction {
	
	public Tanh() {
		this.fnc = new TanhFunction();
		this.derivative = new TanhDerivative();
	}
	
	public static double evaluateTanh(double z) {
		return Math.tanh(z);
	}
	
}
