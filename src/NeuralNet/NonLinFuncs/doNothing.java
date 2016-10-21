package NeuralNet.NonLinFuncs;

import NeuralNet.NonLinAndDerivs.leaveUnchanged;
import NeuralNet.NonLinAndDerivs.returnOne;

// This is for a last layer which does not have a function applied to it
public class doNothing extends NonLinFunction{

	public doNothing() {
		this.fnc = new leaveUnchanged();
		this.derivative = new returnOne();
	}
}
