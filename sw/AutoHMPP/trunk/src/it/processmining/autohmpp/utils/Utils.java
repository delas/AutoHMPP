package it.processmining.autohmpp.utils;

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
}
