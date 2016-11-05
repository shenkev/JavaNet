import NeuralNet.Runners.Runners;

/**
 * 
 */

/**
 * @author kevin
 *
 */
public class main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		// get data
		double[][] Xarr = new double[][]{
		  { -1, -1},
		  { 1, -1 },
		  { -1, 1 },
		  { 1, 1 }
		};	
		
		double[] yarr = new double[] {
			-1,
			1,
			1,
			-1
		};
		
		Runners.Stochastic(Xarr, yarr);
	}
}

