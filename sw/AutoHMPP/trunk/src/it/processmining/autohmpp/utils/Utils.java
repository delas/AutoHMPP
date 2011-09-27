package it.processmining.autohmpp.utils;

import java.io.IOException;

import org.processmining.framework.log.LogFile;
import org.processmining.framework.log.LogFilter;
import org.processmining.framework.log.LogReader;
import org.processmining.framework.log.LogReaderFactory;
import org.processmining.framework.log.filter.DefaultLogFilter;
import org.processmining.framework.models.heuristics.HNSubSet;
import org.processmining.framework.models.heuristics.HeuristicsNet;

public class Utils {

	/**
	 * This method returns the size of the current network, expressed as the
	 * number of edges in the Heuristics Net (so, not distinguishing between AND
	 * and XOR, that in Petri Net representation produces different number of
	 * edges)
	 * 
	 * @param model the model
	 * @return the size of the network
	 */
	public static int calculateNetworkSize(HeuristicsNet model) {
		int connections = 0;
		for (int from = 0; from < model.size(); from++) {
			HNSubSet set = model.getAllElementsOutputSet(from);
			connections += set.size();
		}
		return connections;
	}
	
	
	/**
	 * This method loads a log file into a logfile object 
	 * 
	 * @param inputFile
	 * @return
	 */
	public static LogReader loadLog(String inputFile) {
		LogFile file;
		LogFilter filter = null;

		try {
			if (inputFile.endsWith(".zip")) {
				file = LogFile.getInstance("zip://" + inputFile
						+ "#TestProcess.mxml");
			} else {
				file = LogFile.getInstance(inputFile);
			}
			file.getInputStream();
		} catch (Exception e) {
			file = null;
		}
		if (file == null) {
			try {
				file = LogFile.instantiateEmptyLogFile(inputFile);
				filter = new DefaultLogFilter(DefaultLogFilter.INCLUDE);
			} catch (IOException ex3) {
				System.out
						.println("Cannot create empty Log file: " + inputFile);
				file = null;
				filter = null;
			}
		}

		LogReader log = null;
		try {
			if (file != null) {
				log = LogReaderFactory.createInstance(filter, file);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return log;
	}
}
