package IO;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.text.DecimalFormat;

import robocode.RobocodeFileOutputStream;

public class LUTIO {
	
//	final static String folderPath = "./../workspace/Robocode/results/";
	final static String folderPath = "./results/";
	
	public static void save(String fileName, double[] LUT) {

		   String fullPath = folderPath + fileName;
		   DecimalFormat numberFormat = new DecimalFormat("#.0000");
		   
			PrintStream w = null;
			try {
				w = new PrintStream(new FileOutputStream(fullPath));
				for (int i=0; i < LUT.length; i++) {
					w.println(numberFormat.format(LUT[i]));
				}
			} catch (IOException e) {
				e.printStackTrace();
			} finally {
				w.flush();
				w.close();
			}
	}

	public static double[] load(String fileName, int length) throws IOException {
			
			String fullPath = folderPath + fileName;
			BufferedReader reader = new BufferedReader(new FileReader(fullPath));
			double[] LUT = new double[length];
			
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
			
			return LUT;
	}
	
//	public static double[] load(String fileName, int length) throws IOException {
//		
//		String fullPath = folderPath + fileName;
//		BufferedReader reader = new BufferedReader(new FileReader(fullPath));
//		double[] LUT = new double[length];
//		
//		String line = reader.readLine();
//		try {
//	        int index = 0;
//	        while (line != null) {
//	        	LUT[index] = Double.parseDouble(line);
//	        	line= reader.readLine();
//	        	index++;
//	        }
//		} catch (IOException e) {
//			e.printStackTrace();
//		} finally {
//			reader.close();
//		}
//		
//		return LUT;
//	}
}
