import robocode.Robot;
import robocode.ScannedRobotEvent;

public class TestRobot extends Robot{
	public void run() {
		while (true) {
			ahead(100);
			turnRight(90);
		}
	}
	
	public void onScannedRobot(ScannedRobotEvent e) {
		fire(1);
	}
}
