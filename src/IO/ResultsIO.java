package IO;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.text.DecimalFormat;

import org.la4j.Matrix;

public class ResultsIO {

	public static void writeWins(String fileName, int wins) {
		
		String fullPath = "./../workspace/Robocode/results/" + fileName;
		
		File file= new File (fullPath);
		FileWriter fw;
		if (file.exists())
		{
			try
			{	
				fw = new FileWriter(file, true);//if file exists append to file. Works fine.
			    fw.write(wins + System.getProperty("line.separator"));
			    fw.close();
			}
			catch(IOException ioe)
			{
			    System.err.println("IOException: " + ioe.getMessage());
			}
		}
		else
		{
			try
			{	
			    file.createNewFile();
			    fw = new FileWriter(file);
			    fw.write(wins + System.getProperty("line.separator"));
			    fw.close();
			}
			catch(IOException ioe)
			{
			    System.err.println("IOException: " + ioe.getMessage());
			}

		}
		

	}
	
	public static void save(String fileName, double[] values) {

		   String fullPath = "./../workspace/Robocode/results/" + fileName;
		   DecimalFormat numberFormat = new DecimalFormat("#.0000000000");
		   PrintStream w = null;
		   
			try {
				w = new PrintStream(new FileOutputStream(fullPath));
				
				for(int k=0; k < values.length; k++) {
					
					w.println(numberFormat.format(values[k]));	
				}
			} catch (IOException e) {
				e.printStackTrace();
			} finally {
				w.flush();
				w.close();
			}
		   
	}
}
