package it.processmining.autohmpp;

import it.processmining.autohmpp.search.AutoHMPP;
import it.processmining.autohmpp.utils.Utils;
import it.processmining.hmpp.models.HMPPParameters;

import org.processmining.framework.log.LogFile;
import org.processmining.framework.log.LogReader;

public class Miner {

	public static void main(String[] args) {
		System.out.println("Running...");

		String file = args[0];
		Integer maxPlateauStep = 2;
		Integer threadNumber = 1;
		
		LogReader log = Utils.loadLog(file);
		
		AutoHMPP plugin = new AutoHMPP();
		plugin.setMaxPlateauStep(maxPlateauStep);
		plugin.setNumberOfSearchThread(threadNumber);
		plugin.mine(log);
		
		System.out.println("Complete");
	}
}
