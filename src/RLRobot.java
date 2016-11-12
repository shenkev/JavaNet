import java.awt.Color;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.util.Random;

import IO.ResultsIO;
import robocode.AdvancedRobot;
import robocode.BulletHitEvent;
import robocode.DeathEvent;
import robocode.HitWallEvent;
import robocode.RobocodeFileOutputStream;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

public class RLRobot extends AdvancedRobot {
	// ========================== Hyper Parameters ============================ //
   final double alpha = 0.3;
   final double gamma = 0.975;
   final double epsilon = 0.3;
   
   final double rewardOnWin = 10.0;
	
	// ============================= Policies ================================= //

   
	// ============================ Local Storage ============================= //
    
   static int iter = 0;
   static int previousWins = 0;
   static int previousLosses = 0;
   static final int saveEvery = 100;
   double fieldWidth;
   double fieldHeight;
   Random rand = new Random();
    
   static final int numberOfTotalStateActionParams = 9;
   static final int ownEnergyLevels = 3;
   static final int ownVelocityLevels = 1;
   static final int ownDistanceToEdgeLevels = 5;
   static final int distanceToEnemyLevels = 5;
   static final int enemyEnergyLevels = 3;
   static final int enemyVelocityLevels = 1;
   static final int enemyBearingLevels = 4;
   static final int robotsHeadingLevels = 4;
   static final int numberOfActions = 6;
   static final int numberOfTotalStateActions = ownEnergyLevels*ownVelocityLevels
		   *ownDistanceToEdgeLevels*distanceToEnemyLevels*enemyEnergyLevels
		   *enemyVelocityLevels*enemyBearingLevels*robotsHeadingLevels
		   *numberOfActions;
   
   // For converting state-action to LUT index
   static final int c1 = numberOfTotalStateActions/ownEnergyLevels;
   static final int c2 = c1/ownVelocityLevels;
   static final int c3 = c2/ownDistanceToEdgeLevels;
   static final int c4 = c3/distanceToEnemyLevels;
   static final int c5 = c4/enemyEnergyLevels;
   static final int c6 = c5/enemyVelocityLevels;
   static final int c7 = c6/enemyBearingLevels;
   static final int c8 = c7/robotsHeadingLevels;
   
   // State-Actions
   static double[] LUT = new double[numberOfTotalStateActions];
   int currentStateIndex;
   int currentActionIndex;	// LUT index is the sum of state and action indices
   int previousStateIndex;
   int previousActionIndex;
   double reward;
   
   String saveFileName = "LUT.txt";
   String resultFileName = "results.txt";
   String loadFileName = "LUT.txt";
    
   public void run() {
    	// run at beginning of EACH GAME
	   	if (iter == 0) {
		   	try {
				load(loadFileName);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
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
	   currentActionIndex = epsilonGreedyPolicyChooseAction(previousStateIndex);
	   
	   // always track enemy with gun
	   double absBearing=e.getBearingRadians()+getHeadingRadians();//enemies absolute bearing
	   double latVel=e.getVelocity() * Math.sin(e.getHeadingRadians() -absBearing);//enemies later velocity
	   setTurnRadarLeftRadians(getRadarTurnRemainingRadians());//lock on the radar
	   double gunTurnAmt = robocode.util.Utils.normalRelativeAngle(
			   absBearing- getGunHeadingRadians()+latVel/22);//amount to turn our gun, lead just a little bit
	   setTurnGunRightRadians(gunTurnAmt); //turn our gun
	   
	   // take current action
	   takeAction(currentActionIndex);
	   
	   // observe current state
	   currentStateIndex = returnLUTStateIndex(getEnergy(), getVelocity(), getX(), 
			   getY(), e.getDistance(), e.getEnergy(), e.getVelocity(), 
			   e.getBearing(), getHeading());
	   reward = 0;
	   
	   // update Q(previousState, currentAction)
	   LUT[previousStateIndex + currentActionIndex] = 
			   (1.0 - alpha)*LUT[previousStateIndex + currentActionIndex]
					   + alpha*(reward + gamma*bestQ(currentStateIndex));
	   
	   // update previous state to current state
	   previousStateIndex = currentStateIndex;
   }

   public void onWin(WinEvent event) {

	   LUT[previousStateIndex + currentActionIndex] = 
			   (1.0 - alpha)*LUT[previousStateIndex + currentActionIndex]
					   + alpha*rewardOnWin;
	   iter++;
	   previousWins++;
	   if (iter % saveEvery == 0) {
		   save(saveFileName);
		   // Save results
		   ResultsIO.writeWins(resultFileName, previousWins);
		   previousWins = 0;
		   previousLosses = 0;
	   }
   }
   
   public void onDeath(DeathEvent event) {
	  
	   LUT[previousStateIndex + currentActionIndex] = 
			   (1.0 - alpha)*LUT[previousStateIndex + currentActionIndex]
					   - alpha*rewardOnWin;
	   iter++;
	   previousLosses++;
	   if (iter % saveEvery == 0) {
		   save(saveFileName);
		   // Save results
		   ResultsIO.writeWins(resultFileName, previousWins);
		   previousWins = 0;
		   previousLosses = 0;
	   } 
   }
   
   public void onHitWall(HitWallEvent event) {
	   double hitWallPenalty = 1.0;
	   
	   LUT[previousStateIndex + currentActionIndex] = 
			   (1.0 - alpha)*LUT[previousStateIndex + currentActionIndex]
					   - alpha*hitWallPenalty;
   }
   
   private double bestQ(int stateIndex) {
	   double bestQ = LUT[stateIndex];
	   
	   for (int i=1; i<numberOfActions; i++) {
		   
		   if (LUT[stateIndex + i] > bestQ) {
			   bestQ = LUT[stateIndex + i];
		   }
	   }
	   
	   return bestQ;
   }
   
   public int epsilonGreedyPolicyChooseAction(int stateIndex) {
	   
	   if (rand.nextDouble() < epsilon) {
		   
		   return rand.nextInt(numberOfActions);
	   } else {
		   
		   int bestAction = 0;
		   double bestQ = LUT[stateIndex];
		   
		   for (int i=1; i<numberOfActions; i++) {
			   
			   if (LUT[stateIndex + i] > bestQ) {
				   bestAction = i;
				   bestQ = LUT[stateIndex + i];
			   }
		   }
		   
		   return bestAction;
	   }
   }   
   
   // =================== Helpers for Converting to State ====================== //

   public int returnEnergyLevel(double energy) {
	    	
		double[] thresholds = new double[ownEnergyLevels-1];
		thresholds[0] = 30.0; thresholds[1] = 60.0;
		
		if (energy > thresholds[1]) {
			return 2;
		} else if (energy > thresholds[0]) {
			return 1;
		} else {
			return 0;
		}
   }
    
   public int returnDistToEdgeLevel(double x, double y) {
    	
    	double threshold = 150.0;
    	
    	if (x < threshold) {
    		return 1;
    	} else if (x > (fieldWidth-threshold)) {
    		return 2;
    	} else if (y < threshold) {
    		return 3;
    	} else if (y > (fieldHeight-threshold)) {
    		return 4;
    	} else {
    		return 0;
    	}
   }   
    
   public int returnDistToEnemyLevel(double deltaD) {
	   
	   double[] thresholds = new double[distanceToEnemyLevels-1];
	   thresholds[0] = 50.0; thresholds[1] = 100.0; thresholds[2] = 300.0; thresholds[3] = 500.0;
	   
	   if (deltaD > thresholds[3]) {
		   return 4;
	   } else if (deltaD > thresholds[2]) {
		   return 3;
	   } else if (deltaD > thresholds[1]) {
		   return 2;
	   } else if (deltaD > thresholds[0]) {
		   return 1;
	   } else {
		   return 0;
	   }
   }
   
   public int returnVelocityLevel(double velocity) {
	  
	   return 0;
   }
   
   public int returnBearingLevel(double bearing) {
	   
	   // bearing is -180 <= x < 180
	   if (bearing < 45.0 && bearing >= -45.0) {
		   return 0;
	   } else if (bearing > 45.0 && bearing <= 135.0) {
		   return 1;
	   } else if (bearing >= -135.0 && bearing < -45.0) {
		   return 2;
	   } else {
		   return 3;
	   }
   }
   
   public int returnHeadingLevel(double heading) {
	   
	   // bearing is -180 <= x < 180
	   if (heading < 45.0 && heading >= -45.0) {
		   return 0;
	   } else if (heading > 45.0 && heading <= 135.0) {
		   return 1;
	   } else if (heading >= -135.0 && heading < -45.0) {
		   return 2;
	   } else {
		   return 3;
	   }
   }
   
   public int returnLUTStateIndex(double ownEnergy, double ownVelocity, double ownX,
		   double ownY, double deltaD, double enemyEnergy,
		   double enemyVelocity, double enemyBearing, double ownHeading) {
	   
	   int oE = returnEnergyLevel(ownEnergy);
	   int oV = returnVelocityLevel(ownVelocity);
	   int oDE = returnDistToEdgeLevel(ownX, ownY);
	   int DE = returnDistToEnemyLevel(deltaD);
	   int eE = returnEnergyLevel(enemyEnergy);
	   int eV = returnVelocityLevel(enemyVelocity);
	   int eB = returnBearingLevel(enemyBearing);
	   int rH = returnHeadingLevel(ownHeading);
	   
	   return oE*c1 + oV*c2 + oDE*c3 + DE*c4 + eE*c5 + eV*c6 + eB*c7 + rH*c8;
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
		   setFire(2);
		   break;
		   
	   }
   }
   // =================== Helpers for Logistics ====================== //
   public void save(String fileName) {

	   DecimalFormat numberFormat = new DecimalFormat("#.000");
	   
		PrintStream w = null;
		try {
			w = new PrintStream(new RobocodeFileOutputStream(getDataFile(fileName)));
			for (int i=0; i < numberOfTotalStateActions; i++) {
				w.println(numberFormat.format(LUT[i]));
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			w.flush();
			w.close();
		}
   }
   
   public void load(String fileName) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(getDataFile(fileName)));
		String line = reader.readLine();
		try {
	        int index = 0;
	        while (line != null) {
	        	LUT[index] = Double.parseDouble(line);
	        	line= reader.readLine();
	        	index++;
	        }
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			reader.close();
		}
	}
}