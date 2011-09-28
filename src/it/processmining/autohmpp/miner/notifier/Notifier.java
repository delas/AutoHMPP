package it.processmining.autohmpp.miner.notifier;

/**
 * This interface represents the notification system that receives the updates
 * from the search threads
 * 
 * @author Andrea Burattin
 * @version 0.1
 */
public interface Notifier {
	
	/**
	 * Enumeration of the possible steps of the algorithm
	 */
	public enum STEPS {
		PARAMETER_DISCRETIZATION,
		SEARCH_FIRST_CLUSTER,
		SEARCH_SECOND_CLUSTER
	};
	
	
	/**
	 * Method called when a step starts
	 * 
	 * @param threadName the name of the given thread
	 * @param step the step that has just begun
	 */
	public void stepStarts(String threadName, STEPS step);
	
	
	/**
	 * Method called when a step finishes
	 * 
	 * @param threadName the name of the given thread
	 * @param step the step that has just finished
	 */
	public void stepEnds(String threadName, STEPS step);
	
	
	/**
	 * Method called every time a new step is performed, towards the solution
	 * 
	 * @param threadName the name of the given thread
	 * @param stepNo the step counter
	 * @param position a representation of the new position
	 * @param cost the cost of the new solution
	 */
	public void notifyStep(String threadName, Integer stepNo, String position, Double cost);
	
	
	/**
	 * Method called every time, given a particular step, all the possible directions are checked
	 * 
	 * @param threadName the name of the given thread
	 * @param stepNo the step counter
	 * @param position a representation of the new position
	 * @param direction the direction, given a particular position
	 * @param directionCost the cost of the new solution
	 */
	public void notifyDirection(String threadName, Integer stepNo, String position, String direction, Double directionCost);
}
