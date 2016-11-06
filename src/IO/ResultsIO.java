package IO;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

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
}
