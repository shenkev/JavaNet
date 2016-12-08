
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;

import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.DenseMatrix;

import IO.LUTIO;
import IO.WeightIO;
import NeuralNet.NeuralNet;

public class offlineTraining {
	
	// Random seed
	static Random rand = new Random(800);
	
	// Train data
	static Matrix X;
	static Matrix y;
	static String folderPath = "./results/NNWeights/";
	
	static String fileName = "RLtrainLUT.txt";

	public static void main(String[] args) {
		
		loadOfflineDat();
		
//		int batchSize = X.rows();
		int batchSize = 1;
		NeuralNet nn = new NeuralNet(NNRLRobot.numberOfFeatures, batchSize,
				NNRLRobot.noLayers, NNRLRobot.layerDims, NNRLRobot.nonLinFunction,
				NNRLRobot.outputFunction, NNRLRobot.lossFunc, NNRLRobot.optimizer,
				800);
		
//   		try {
//   			// load weights
//			nn.setW(WeightIO.load(folderPath, NNRLRobot.loadWfolder, NNRLRobot.dims));
//			
//			// load biases
//			Matrix[] mb = WeightIO.load(folderPath, NNRLRobot.loadbfolder, NNRLRobot.dimsb);
//			Vector[] bb = new Vector[mb.length];
//			for(int i=0; i < bb.length; i++) {
//				bb[i] = mb[i].toRowVector();
//			}
//			nn.setb(bb);
//			
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
		
		Random rand = new Random();
		// training params
		int iter = 1591*400;
		int printPer = 1591;
		int convergedIteration = 0;
		boolean converged = false;
		
		double[] losses = new double[iter];
		double lossSum = 0;
		
		for ( int i = 0; i < iter; i++ ) {
			
			 int j = rand.nextInt(X.rows());
			 double loss = nn.runOnePass(X.getRow(j).toRowMatrix(), y.getRow(j).toRowMatrix());
			 losses[i] = loss;
			 lossSum = lossSum + loss;
			
			 if ( loss < 0.05 && converged == false ) {
				 converged = true;
				 convergedIteration = i;
			 }
			 
			 if ( i % printPer == 0 ) {
//				 System.out.println( "Loss for iteration " + i + " is: " + lossSum );
				 System.out.println(lossSum);
				 lossSum = 0;
			 }
		}
//		System.out.println(nn.predict(X).subtract(y));

//		System.out.println("Converged at iteration: " + convergedIteration);	
		   // save weights
		   WeightIO.save(folderPath, NNRLRobot.loadWfolder,
				  NNRLRobot.saveWName, nn.getW());
		   
		   // save biases
		   Vector[] bb = nn.getb();
		   Matrix[] mb = new Matrix[bb.length];
		   for(int i=0; i < mb.length; i++) {
		   	mb[i] = bb[i].toRowMatrix();
		   }
		   WeightIO.save(folderPath, NNRLRobot.loadbfolder,
				   NNRLRobot.savebName, mb);
		
		
		
	}

	public static Object[] loadOfflineDat() {
		
		// Load LUT
		double[] LUT = null;
		try {
			String test = Paths.get(".").toAbsolutePath().normalize().toString();
			LUT = LUTIO.load(fileName, RLRobot.numberOfTotalStateActions);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		double[][] Xarr = new double[LUT.length/RLRobot.numberOfActions][NNRLRobot.numberOfTotalStateParams];
		double[][] yarr = new double[LUT.length/RLRobot.numberOfActions][NNRLRobot.numberOfActions];
		
		// Convert LUT entries to states and one-hot action rewards
		int state = -1;
		double total = 100;
		for(int i=0; i < LUT.length; i++) {
			
			int iter = i;
		    int oE = iter/RLRobot.c1;
		    iter = iter%RLRobot.c1;
		    int oV = iter/RLRobot.c2;
		    iter = iter%RLRobot.c2;
		    int oDE = iter/RLRobot.c3;
		    iter = iter%RLRobot.c3;
		    int DE = iter/RLRobot.c4;
		    iter = iter%RLRobot.c4;
		    int eE = iter/RLRobot.c5;
		    iter = iter%RLRobot.c5;
		    int eV = iter/RLRobot.c6;
		    iter = iter%RLRobot.c6;
		    int eB = iter/RLRobot.c7;
		    iter = iter%RLRobot.c7;
			int rH = iter/RLRobot.c8;
			iter = iter%RLRobot.c8;
			int act = iter;
			
			double ownEnergy = returnEnergy(oE);
			double ownVelocity = returnVelocity(oV);
			double[] ownPosition = returnCoord(oDE);
			double ownX = ownPosition[0];
			double ownY = ownPosition[1];
			double ownHeading = returnRadians(rH);
			double enemyEnergy = returnEnergy(eE);
			double enemyBearing = returnRadians(eB);
			double enemyDistance = returnDistToEn(DE);
			double enemyVelocity = returnVelocity(eV);
			double enemyHeading = 0;

			if (act == 0) {
				// we overwrite state of it is all 0s
				if (total > 0.5) {
					state++;
				}
					
				total = 0;
				Xarr[state] = returnState(ownEnergy, ownVelocity, ownX, ownY,
						ownHeading, enemyEnergy, enemyBearing, enemyDistance,
						enemyVelocity, enemyHeading);
			}
			
			total = total + Math.abs(LUT[i]);
			yarr[state][act] = LUT[i];
		}
		
		Xarr = Arrays.copyOfRange(Xarr, 0, state);
		yarr = Arrays.copyOfRange(yarr, 0, state);
		
		X = DenseMatrix.from2DArray(Xarr);
		y = DenseMatrix.from2DArray(yarr);
		
		return new Object[]{Xarr, yarr};
	}
	
	// These functions are expected to match those in RLRobot.java
//	public static int returnAction(int level) {
//		
//		if (level == 0) {
//			// some probability move forward 0, 45, -45 degrees
//			double temp = rand.nextDouble();
//			if (temp < 2.0/3.0) {
//				return 0;
//			} else if (temp > 2.5/3.0) {
//				return 1;
//			} else {
//				return 7;
//			}
//			
//		} else if (level == 1) {
//			// some probability move forward -180, 135, -135 degrees
//			double temp = rand.nextDouble();
//			if (temp < 2.0/3.0) {
//				return 4;
//			} else if (temp > 2.5/3.0) {
//				return 5;
//			} else {
//				return 3;
//			}
//			
//		} else if (level == 2) {
//			// some probability move forward -90, -45, -135 degrees
//			double temp = rand.nextDouble();
//			if (temp < 2.0/3.0) {
//				return 6;
//			} else if (temp > 2.5/3.0) {
//				return 5;
//			} else {
//				return 7;
//			}
//			
//		} else if (level == 3) {
//			// some probability move forward 90, 45, 135 degrees
//			double temp = rand.nextDouble();
//			if (temp < 2.0/3.0) {
//				return 2;
//			} else if (temp > 2.5/3.0) {
//				return 1;
//			} else {
//				return 3;
//			}
//			
//		} else if (level == 4) {
//			return 8;
//		} else {
//			return 9;
//		}
//	}
	
	public static double returnEnergy(int level) {
		
		if (level == 0) {
			return rand.nextDouble()*30.0;
		} else if (level == 1) {
			return 30.0 + rand.nextDouble()*30.0;
		} else {
			return 60.0 + rand.nextDouble()*50.0;
		}
		
	}
	
	public static double returnVelocity(int level) {
		
		return rand.nextDouble()*5.0;
	}
	
	public static double returnDistToEn(int level) {
		
		if (level == 0) {
			return rand.nextDouble()*50.0;
		} else if (level == 1) {
			return 50.0 + rand.nextDouble()*50.0;
		} else if (level == 2) {
			return 100.0 + rand.nextDouble()*200.0;
		} else if (level == 3) {
			return 300.0 + rand.nextDouble()*200.0;
		} else {
			return 500.0 + rand.nextDouble()*500.0;
		}
	}
	
	public static double[] returnCoord(int level) {
		
		double[] xy = new double[2];
		
		if (level == 1) {
			xy[0] = rand.nextDouble()*150.0;
			xy[1] = rand.nextDouble()*600.0;
		} else if (level == 2) {
			xy[0] = 650.0 + rand.nextDouble()*150.0;
			xy[1] = rand.nextDouble()*600.0;
		} else if (level == 3) {
			xy[0] = rand.nextDouble()*800.0;
			xy[1] = rand.nextDouble()*150.0;
		} else if (level == 4) {
			xy[0] = rand.nextDouble()*800.0;
			xy[1] = 450.0 + rand.nextDouble()*150.0;
		} else {
			xy[0] = 150.0 + rand.nextDouble()*500.0;
			xy[1] = 150.0 + rand.nextDouble()*300.0;
		}
		
		return xy;
	}
	
	public static double returnRadians(int level) {
		
		if (level == 0) {
			return 0.7854 - 2*rand.nextDouble()*0.7854;
		} else if (level == 1) {
			return 0.7854 + 2*rand.nextDouble()*0.7854;
		} else if (level == 2) {
			return -0.7854 - 2*rand.nextDouble()*0.7854;
		} else {
			return 2.3562 + 2*rand.nextDouble()*0.7854;
		}
		
	}
	
	// This should match NNRLRobot.java
	public static double[] returnState(double ownEnergy, double ownVelocity, double ownX,
			   double ownY, double ownHeading, double enemyEnergy,
			   double enemyBearing, double enemyDistance,
			   double enemyVelocity, double enemyHeading) {
		   
		   return new double[] {
				   
				   (ownEnergy-NNRLRobot.energyMean)/NNRLRobot.energyScale,
				   (ownVelocity-NNRLRobot.velocityMean)/NNRLRobot.velocityScale,
				   (ownX - (800.0/2.0))/240.0, 
				   (ownY - (600.0/2.0))/180.0,
				   1.4*Math.cos(ownHeading), 1.4*Math.sin(ownHeading),
				   (enemyEnergy-NNRLRobot.energyMean)/NNRLRobot.energyScale,
				   1.4*Math.cos(enemyBearing), 1.4*Math.sin(enemyBearing),
				   (enemyDistance-200.0)/210.0,
				   (enemyVelocity-NNRLRobot.velocityMean)/NNRLRobot.velocityScale,
				   1.4*Math.cos(enemyHeading), 1.4*Math.sin(enemyHeading)
				   
		   };
	   }
}
