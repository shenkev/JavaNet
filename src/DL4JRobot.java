import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import IO.ResultsIO;
import robocode.AdvancedRobot;
import robocode.BulletHitEvent;
import robocode.DeathEvent;
import robocode.HitWallEvent;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

public class DL4JRobot extends AdvancedRobot {

	// ============================ Local Storage ============================= //
	   static int iter = 0;
	   static int previousWins = 0;
	   static int previousLosses = 0;
	   static final int saveWeightsEvery = 1000;
	   static final int saveResultsEvery = 100;
	   static String folderPath = "./../workspace/Robocode/results/offline/";
	   static String saveName = "robotModel.zip";
	   static String loadName = "MyMultiLayerNetwork.zip";
	   static String resultFileName = "results.txt";
		
		// ========================== RL Hyper Parameters ============================ //
	   final double gamma = 0.975;
	   final double epsilon = 0.1;	// should start at 1 and decrement to 0.1
	   
	   final double rewardOnWin = 7.0;
	   
		// ========================== State Representation ========================== //
	   Random rand = new Random();
	    
	   static final int numberOfTotalStateParams = 13;
	   static final int numberOfActions = 6;
	   static final int numberOfFeatures = numberOfTotalStateParams;
	   
	   // indexes of features
	   static final int ownEnergy = 0;
	   static final int ownVelocity = 1;
	   static final int ownX = 2;
	   static final int ownY = 3;
	   static final int ownHeadingCos = 4;
	   static final int ownHeadingSin = 5;
	   static final int enemyEnergy = 6;
	   static final int enemyBearingCos = 7;
	   static final int enemyBearingSin = 8;
	   static final int enemyDistance= 9;
	   static final int enemyVelocity = 10;
	   static final int enemyHeadingCos = 11;
	   static final int enemyHeadingSin = 12;
	   
	   // constants for making variance = 1 for input features
	   static final double energyScale = 26.0;
	   static final double energyMean = 40.0;
	   static final double velocityScale = 1.4;
	   static final double velocityMean = 2.5;
	   static final double xScale = 240.0;
	   static final double yScale = 180.0;
	   static final double distScale = 210.0;
	   static final double angleScale = 1.4;
	   
	   double fieldWidth;
	   double fieldHeight;
	   
	   // State-Actions
	   INDArray currentState;
	   int currentAction;
	   // Initialize state
	   static final double[][] prevStateArr = new double[][]{
			   {100/energyScale},
			   {0},
			   {0}, {0},
			   {1}, {0},
			   {100/energyScale},
			   {1}, {0},
			   {300/800}, {0},
			   {1}, {0}
	   };
	   INDArray previousState = Nd4j.create(prevStateArr);
	   
	   // Batching
	   static final int batchSize = 18;
	   static int batchIndex = 0;
	   INDArray batchState;
	   INDArray batchLabels;
	   
	   double reward;
	   
		// ========================== Neural Network Setup ========================== //
	   static MultiLayerNetwork nn;

	   // ========================== Start of Robot Code ========================== //
	   public void run() {
	   	// run at beginning of EACH GAME
		   	if (iter == 0) { // only do this during the first game
		   		try {
		   			// load model
		   			nn = ModelSerializer.restoreMultiLayerNetwork("./results/offline/MyMultiLayerNetwork.zip");
					
				} catch (IOException e) {

					e.printStackTrace();
				}
		   		
		   		if (nn == null) {
		   			throw new IllegalArgumentException("Couldn't find the pretrained model.");
		   		}
		   		
	   			nn.init();
		   	}
		   
		    fieldWidth = getBattleFieldWidth();
		    fieldHeight = getBattleFieldHeight();
	        
			setAdjustRadarForRobotTurn(true);//keep the radar still while we turn
			setBodyColor(Color.red);
			setGunColor(Color.white);
			setRadarColor(Color.white);
			setScanColor(Color.white);
			setBulletColor(Color.red);
			setAdjustGunForRobotTurn(true); // Keep the gun still when we turn
			turnRadarRightRadians(Double.POSITIVE_INFINITY);//keep turning radar right
			
	    	while (true) {

	        }
	   }
	 
	   // ============================ Q-Learning ============================= //
	   public void onScannedRobot(ScannedRobotEvent e) {
	       // choose current action from previous state using epsilon policy
		   List<INDArray> QforPrevStateActionsMatrix = nn.feedForward(previousState, false);
		   currentAction = epsilonGreedyPolicyChooseAction(
				   QforPrevStateActionsMatrix.get(0));
		   
		   // always track enemy with gun
		   double absBearing=e.getBearingRadians()+getHeadingRadians();//enemies absolute bearing
		   double latVel=e.getVelocity() * Math.sin(e.getHeadingRadians() -absBearing);//enemies later velocity
		   setTurnRadarLeftRadians(getRadarTurnRemainingRadians());//lock on the radar
		   double gunTurnAmt = robocode.util.Utils.normalRelativeAngle(absBearing- getGunHeadingRadians()+latVel/22);//amount to turn our gun, lead just a little bit
		   setTurnGunRightRadians(gunTurnAmt); //turn our gun
		   
		   // take current action
		   takeAction(currentAction);
		   
		   // observe current state
		   double[][] currentStateArr = returnState(getEnergy(), getVelocity(),
				   getX(), getY(),
				   getHeadingRadians(), e.getEnergy(), e.getBearingRadians(),
				   e.getDistance(), e.getVelocity(), e.getHeadingRadians());
		   currentState = Nd4j.create(currentStateArr);
		   
		   reward = 0;
		   
		   // find maxQ for s'
		   List<INDArray> QforCurrStateActionsMatrix = nn.feedForward(currentState, false);
		   double maxQ = bestQ(
				   QforCurrStateActionsMatrix.get(0));
		   double target = reward + gamma*maxQ;
		   
		   // QforPrevStateActionsMatrix is now the "truth" or target
		   INDArray targINDArr = QforPrevStateActionsMatrix.get(0).putScalar(currentAction, target);
		   
		   // train Q(previousState, currentAction) via backpropagation
		   batchState.putRow(batchIndex, previousState);
		   batchLabels.putRow(batchIndex, targINDArr);
		   
		   // hold on until we have a batch
		   if (batchIndex == batchSize - 1) {
			   nn.fit(batchState, batchLabels);
			   batchIndex = 0;
		   } else {
			   batchIndex++;
		   }
		   
		   // update previous state to current state
		   previousState = currentState;

	   }

	   public void onWin(WinEvent event) {

		   // update previous action-state with reward only
		   List<INDArray> QforPrevStateActionsMatrix = nn.feedForward(previousState, false);
		   INDArray targINDArr = QforPrevStateActionsMatrix.get(0).putScalar(currentAction, rewardOnWin);
		   
		   // Train what we have
		   batchState.putRow(batchIndex, previousState);
		   batchLabels.putRow(batchIndex, targINDArr);
		   nn.fit(batchState, batchLabels);
		   batchIndex = 0;
		   		   
		   // save
		   iter++;
		   previousWins++;
		   // save results
		   if (iter % saveResultsEvery == 0) {
			   ResultsIO.writeWins(resultFileName, previousWins);
			   previousWins = 0;
			   previousLosses = 0;
		   }
		   // save Weights
		   if (iter % saveWeightsEvery == 0) {
			   
			   File locationToSave = new File(folderPath + saveName);      
			   // Where to save the network. Note: the file is in .zip format - can be opened externally
			   boolean saveUpdater = true;                                     
			   // Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train
			   // your network more in the future
			   try {
				ModelSerializer.writeModel(nn, locationToSave, saveUpdater);
				} catch (IOException e) {
					e.printStackTrace();
				}

		   }
	   }
	   
	   public void onDeath(DeathEvent event) {
		  
		   // update previous action-state with reward only
		   List<INDArray> QforPrevStateActionsMatrix = nn.feedForward(previousState, false);
		   INDArray targINDArr = QforPrevStateActionsMatrix
				   .get(0).putScalar(currentAction, (-1.0*rewardOnWin));
		   
		   // Train what we have
		   batchState.putRow(batchIndex, previousState);
		   batchLabels.putRow(batchIndex, targINDArr);
		   nn.fit(batchState, batchLabels);
		   batchIndex = 0;
		   		   
		   // save
		   iter++;
		   previousLosses++;
		   // save results
		   if (iter % saveResultsEvery == 0) {
			   ResultsIO.writeWins(resultFileName, previousWins);
			   previousWins = 0;
			   previousLosses = 0;
		   }
		   // save Weights
		   if (iter % saveWeightsEvery == 0) {
			   
			   File locationToSave = new File(folderPath + saveName);      
			   // Where to save the network. Note: the file is in .zip format - can be opened externally
			   boolean saveUpdater = true;                                     
			   // Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train
			   // your network more in the future
			   try {
				ModelSerializer.writeModel(nn, locationToSave, saveUpdater);
				} catch (IOException e) {
					e.printStackTrace();
				}
		   }
	   }
	   
	   public void onBulletHit(BulletHitEvent event) {
		   
	   }
	   
	   public void onHitWall(HitWallEvent event) {
		   double hitWallPenalty = 0.1*rewardOnWin;
		   
		   // update previous action-state with reward only
		   List<INDArray> QforPrevStateActionsMatrix = nn.feedForward(previousState, false);
		   INDArray targINDArr = QforPrevStateActionsMatrix
				   .get(0).putScalar(currentAction, (-1.0*hitWallPenalty));
		   
		   // Train what we have, shortcircuit the batching to keep it simple
		   batchState.putRow(batchIndex, previousState);
		   batchLabels.putRow(batchIndex, targINDArr);
		   nn.fit(batchState, batchLabels);
		   batchIndex = 0;
		   
	   }
	   
	   private double bestQ(INDArray QforActions) {
		   
		   double bestQ = QforActions.getDouble(0);
		   for (int i=1; i<numberOfActions; i++) {
			   if (QforActions.getDouble(i) > bestQ) {
				   bestQ = QforActions.getDouble(i);
			   }
		   }
		   
		   return bestQ;
	   }
	   
	   public int epsilonGreedyPolicyChooseAction(INDArray QforActions) {
		   
		   if (rand.nextDouble() < epsilon) {
			   
			   return rand.nextInt(numberOfActions);
		   } else {
			  
			   int bestAction = 0;
			   double bestQ = QforActions.getDouble(0);
			   
			   for (int i=1; i<numberOfActions; i++) {
				   if (QforActions.getDouble(i) > bestQ) {
					   bestAction = i;
					   bestQ = QforActions.getDouble(i);
				   }
			   }
			   
			   return bestAction;
		   }
	   }   
	   
	   // =================== Helpers for Converting to State ====================== //
	   public double[][] returnState(double ownEnergy, double ownVelocity, double ownX,
			   double ownY, double ownHeading, double enemyEnergy,
			   double enemyBearing, double enemyDistance,
			   double enemyVelocity, double enemyHeading) {
		   
		   double[][] arr = new double[1][numberOfTotalStateParams];
		   arr[0] = new double[] {
				   
				   (ownEnergy-energyMean)/energyScale,
				   (ownVelocity-velocityMean)/velocityScale,
				   (ownX - (800.0/2.0))/xScale, 
				   (ownY - (600.0/2.0))/yScale,
				   angleScale*Math.cos(ownHeading), angleScale*Math.sin(ownHeading),
				   (enemyEnergy-energyMean)/energyScale,
				   angleScale*Math.cos(enemyBearing), angleScale*Math.sin(enemyBearing),
				   (enemyDistance-200.0)/distScale,
				   (enemyVelocity-velocityMean)/velocityScale,
				   angleScale*Math.cos(enemyHeading), angleScale*Math.sin(enemyHeading)
				   
		   };
		   
		   return arr;
	   }
	   
	   public void takeAction(int action) {
		   
		   double moveStep = 50.0;
		   
		   
		   switch(action) {
		   case 0:
			   setAhead(moveStep);
			   break;
			   
		   case 1:
			   setBack(moveStep);
			   break;
			   
		   case 2:
			   setTurnLeft(90.0);
			   setAhead(moveStep);
			   break;
			   
		   case 3:
			   setTurnRight(90.0);
			   setAhead(moveStep);
			   break;
			   
		   case 4:
			   setFire(1);
			   break;
			   
		   case 5:
			   setFire(1.5);
			   break;
			   
		   }
	   }
	
}
