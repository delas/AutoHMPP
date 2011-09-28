package it.processmining.autohmpp.miner.notifier;

import java.io.PrintStream;

/**
 * Class for the notification of the messages in a general print stream
 * 
 * @author Andrea Burattin
 * @version 0.1
 */
public class PrintStreamNotifier implements Notifier {
	
	private PrintStream out;
	
	
	/**
	 * Class constructor
	 * 
	 * @param out the output print stream
	 */
	public PrintStreamNotifier(PrintStream out) {
		this.out = out;
	}
	

	@Override
	public void stepStarts(String threadName, STEPS step) {
		String message = "Thread: "+ threadName +"\tStarted ";
		switch (step) {
		case PARAMETER_DISCRETIZATION:
			message += "parameter discretization";
			break;
		case SEARCH_FIRST_CLUSTER:
			message += "search in the first cluster";
			break;
		case SEARCH_SECOND_CLUSTER:
			message += "search in the second cluster";
			break;
		}
		out.println(message);
		out.flush();
	}
	

	@Override
	public void stepEnds(String threadName, STEPS step) {
		String message = "Thread: "+ threadName +"\tFinished ";
		switch (step) {
		case PARAMETER_DISCRETIZATION:
			message += "parameter discretization";
			break;
		case SEARCH_FIRST_CLUSTER:
			message += "search in the first cluster";
			break;
		case SEARCH_SECOND_CLUSTER:
			message += "search in the second cluster";
			break;
		}
		out.println(message);
		out.flush();
	}
	

	@Override
	public void notifyStep(String threadName, Integer stepNo, String position, Double cost) {
		out.println("Thread: "+ threadName +"\tStep: "+ stepNo +"\tPosition: " + position +"\tCost: " + cost);
		out.flush();
	}


	@Override
	public void notifyDirection(String threadName, Integer stepNo, String position, String direction, Double directionCost) {
		out.println("Thread: "+ threadName +"\tStep: "+ stepNo +"\tPosition: " + position +"\tDirection: "+ direction +"\tCost: " + directionCost);
		out.flush();
	}

}
