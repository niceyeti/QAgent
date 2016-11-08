#include "Q2Agent.hpp"

/*
TODO's for experimentation:
	-different ann architectures: first, single net instead of a network for each action. second, recurrent nets.
	-offline learning
	-different update schemes:
		1) try updating the agent only when it either dies or reaches the goal (instantaneous, strong rewards)
		2) other update strategies
	-other learning methods? 
		online: are there any outside neural nets? rbf nets?
		offline: regression, polynomial regression, etc

*/

kvector::kvector(const vector<double>& state, double reward, char label)
{
	xs = state;
	r = reward;
	alpha = label;
}

Experience::Experience()
{
	BatchedState.resize(STATE_DIMENSION);
	QTarget = 0.0;
	QEstimate = 0.0;
	PerformedAction = ACTION_DOWN;
}

Q2Agent::Q2Agent(int initX, int initY)
{
	//init the agent
	agent.x = initX;
	agent.y = initY;
	agent.isAlive = true;
	agent.sufferedCollision = false;
	agent.xVelocity = 0;
	agent.yVelocity = 0;
	CurrentAction = ACTION_UP;

	_totalEpisodes = 0;
	_lastEpochCollisionRate = 0;
	_lastEpochActionCount = 0;
	_episodeCount = 0;
	_epochCount = 0;
	_epochReward = 0;
	_lastEpochReward = 0;
	_repetitionCounter = 0;
	EpochGoalCount = 0;
	EpochActionCount = 0;
	EpochCollisionCount = 0;
	GoalResetThreshold = 1;

	//set _eta value, the q-learning learning rate
	_eta = 0.05;
	_gamma = 0.9;
	GoalResetThreshold = 1;
	_t = 0; //time series index
	_currentActionValues.resize(NUM_ACTIONS); //for caching the q-values per action, instead of re-calling Classify() on the net, per action
	//init the neural networks, one for each action
	_qNet.BuildNet(2, STATE_DIMENSION, STATE_DIMENSION, 1); //this is just generic, for testing;
	//set outputs to linear, since q-values are linear in some range after convergence like [-10.12-8.34]
	_qNet.SetHiddenLayerFunction(TANH);
	_qNet.SetOutputLayerFunction(LINEAR);
	_qNet.InitializeWeights();
	_qNet.SetEta(_eta);
	//TODO: momentum is good in general, but I'm not sure the effect in this context. In general it speeds training and helps escape local minima.
	_qNet.SetMomentum(0.2);
	//set the regularization term
	_qNet.SetWeightDecay(0.0001);

	//init the reward-approximation neural net (this is purely experimental, may not be used in all learning paradigms)
	_rewardApproximator.BuildNet(2, STATE_DIMENSION, STATE_DIMENSION, 1); //this is just generic, for testing;
	_rewardApproximator.SetHiddenLayerFunction(TANH);
	_rewardApproximator.SetOutputLayerFunction(LINEAR);
	_rewardApproximator.InitializeWeights();
	_rewardApproximator.SetEta(0.1);
	//TODO: momentum is good in general, but I'm not sure the effect in this context. In general it speeds training and helps escape local minima.
	_rewardApproximator.SetMomentum(0.2);
	//set the regularization term
	_rewardApproximator.SetWeightDecay(0.001);

	//init the state history; only t and t+1 for now
	_stateHistory.resize(2);
	for(int i = 0; i < _stateHistory.size(); i++){
		_stateHistory[i].resize(NUM_ACTIONS); //each neural net has its own state vector at each time step
		for(int j = 0; j < _stateHistory[i].size(); j++){
			_stateHistory[i][j].resize(STATE_DIMENSION, 0.0);
		}
	}

	//configure the location hisorian to hold up to k previous locations
	_recentLocations.resize(NUM_MEMORIZED_LOCATIONS, std::pair<double,double>(0,0));
	_locationRingIndex = 0;
	_locationEma.first = 0.0;
	_locationEma.second = 0.0;
	_locationAvg.first = 0.0;
	_locationAvg.second = 0.0;

	//set up the expectation maximization neurons
	_alphaNeurons.insert( pair<char,Neuron>(ALPHA_GOAL_REACHED, Neuron(STATE_DIMENSION + 1, LOGISTIC)));
	_alphaNeurons.insert( pair<char,Neuron>(ALPHA_REPETITION, Neuron(STATE_DIMENSION + 1, LOGISTIC)));
	_alphaNeurons.insert( pair<char,Neuron>(ALPHA_COLLISION, Neuron(STATE_DIMENSION + 1, LOGISTIC)));
	//randomize the neurons' initial weights
	for(auto it = _alphaNeurons.begin(); it != _alphaNeurons.end(); ++it){
		//it->second.AssignRandomWeights(1.0,-1.0);
		it->second.AssignUniformWeights(0.0); //effectively inits all uniforms to output equal binary event probability 0.50
	}

	//set up the various output files
	_historyFilePath = "./history.txt";
	_outputFile.open(_historyFilePath,ios::out);
	if(!_outputFile.is_open()){
		cout << "ERROR could not open history.txt file" << endl;
	}

	_prototypeFile.open("prototypes.csv",ios::out);
	if(!_prototypeFile.is_open()){
		cout << "ERROR could not open prototype.csv file" << endl;
	}

	_kVectorFile.open("kvectors.csv", ios::out);
	if(!_kVectorFile.is_open()){
		cout << "ERROR could not open kvectors.csv file" << endl;
	}

	_rewardParamsFile.open("rewardParams.csv", ios::out);
	if(!_rewardParamsFile.is_open()){
		cout << "ERROR could not open rewardParams.csv file" << endl;
	}

	//write the headers. not using csv for now, oh well...
	//_outputFile << "<comma-delimited state values>,qTarget,qEstimate,ActionString" << endl;
}

//yada yada
Q2Agent::~Q2Agent()
{
	if(_outputFile.is_open()){
		_outputFile.close();
	}
}

//returns the t-1th state for the given action
const vector<double>& Q2Agent::_getPreviousState(Action action)
{
	if(_t == 0){
		return _stateHistory[ _stateHistory.size() - 1 ][(int)action];
	}
	return _stateHistory[_t-1][(int)action];
}

const vector<double>& Q2Agent::_getCurrentState(Action action)
{
	return _stateHistory[_t][(int)action];
}

/*
Updates the estimate of the mean location over the last k-visited locations. This gives the agent
a mechanism to aproximate places it has been recently. Here I set both an arithmetic average value
and an exponential moving average. 
*/
void Q2Agent::_updateLocationMemory()
{
	//update recent location knowledge: locations and the moving average of previous locations
	_locationRingIndex = (_locationRingIndex + 1) % _recentLocations.size();
	_recentLocations[_locationRingIndex].first = agent.x;
	_recentLocations[_locationRingIndex].second = agent.y;
	//update the exp moving average of previous locations; note this is a recursive form
	double alpha = 0.5;
	_locationEma.first = alpha * _recentLocations[max(_locationRingIndex - 1, 0)].first + (1 - alpha) * _locationEma.first;
	_locationEma.second = alpha * _recentLocations[max(_locationRingIndex - 1, 0)].second + (1 - alpha) * _locationEma.second;
	//_locationEma.first = alpha * _recentLocations[max(_locationRingIndex - 1, 0)].first + (1 - alpha) * _recentLocations[ max(_locationRingIndex - 2, 0)].first;
	//_locationEma.second = alpha * _recentLocations[max(_locationRingIndex - 1, 0)].second + (1 - alpha) * _recentLocations[ max(_locationRingIndex - 2, 0)].second;
	//update the arithmetic average over previous k locations, giving the centroid of previous k locations
	_locationAvg.first = _locationAvg.second = 0.0;
	for(int i = 0; i < _recentLocations.size(); i++){
		if(i != _locationRingIndex){ //excludes current location from estimate of past locations
			_locationAvg.first += _recentLocations[i].first;
			_locationAvg.second += _recentLocations[i].second;
		}
	}
	_locationAvg.first /= ((double)_recentLocations.size() - 1.0); //minus one since the current location is excluded
	_locationAvg.second /= ((double)_recentLocations.size() - 1.0);
}

void Q2Agent::_updateCurrentActionStates(const World* world, const vector<Missile>& missiles)
{
	//update the time step; this is just an index for _stateHistory for now
	_t = (_t + 1) % (int)_stateHistory.size();
	
	//update agent's estimate of its recent locations
	_updateLocationMemory();
	
	//derive current state estimate, per each possible action
	_deriveActionStates(world, missiles, _stateHistory[_t]);

	//experimental: normalize the state vector such that the neural net inputs are effectively zero-mean
	for(int action = 0; action < _stateHistory[_t].size(); action++){
		_normalizeStateVector(world, _stateHistory[_t][action]);
	}

	//PrintCurrentStateEstimates();
}

//Each action has a state estimate (the state that would be a consequence of taking the action). This prints these.
void Q2Agent::PrintCurrentStateEstimates()
{
	for(int action = 0; action < _stateHistory[_t].size(); action++){
		cout << "state " << GetActionStr(action) << ":" << endl;
		for(int i = 0; i < _getCurrentState((Action)action).size(); i++){
			cout << _getCurrentState((Action)action)[i] << " ";
		}
		cout << endl;
	}
}


/*
Normalizes the state vector.

This is experimental.

TODO: Is this desirable over zero-mean? The only reason I'm keeping this around is for testing.
*/
void Q2Agent::_normalizeStateVector(const World* world, vector<double>& state)
{
	//NOTE: NO NEED TO NORMALIZE state[SA_LOCATION_VISIT_COUNT] since it is discrete 0.0 or 1.0

	//both cos-sim metrics just get shiftd down by one to give them range -2.0 (opposite some other point), to 0.0 (in direction of some point)
	state[SA_GOAL_COSINE] = state[SA_GOAL_COSINE] - 1.0;
	//the direction maximizing this cossim is if agent goes in the direction of its previous location. since this is
	//undesirable, i reverse the cossim metric, such that -2.0 is the direction maximizing similarity to previous locations, and 0.0 is best
	state[SA_RECENT_LOCATION_COSINE] = -2.0 - (state[SA_RECENT_LOCATION_COSINE] - 1.0);
	//note cosine attribute is not normalized, since cosine is inherently normalized
	//cout << "collision proximity: " << 	state[SA_COLLISION_PROXIMITY] << endl;
	//collision proximity is scaled to [-2.0,0.0], where -2.0 is worst (immediate collision) and 0.0 is greatest distance to an obstacle
	state[SA_COLLISION_PROXIMITY] = 0.0 - (2.0 * ((double)MAX_COLLISION_PROXIMITY_RANGE - state[SA_COLLISION_PROXIMITY])) / (double)MAX_COLLISION_PROXIMITY_RANGE;
	//cout << "AFTER: " << state[SA_COLLISION_PROXIMITY] << endl;
	//goal distance is scaled to range [-1.0,1.0] with -1.0 being worst (greatest distance) and 1.0 being the best
	//state[SA_GOAL_DIST] = 0.0 - (2.0 * state[SA_GOAL_DIST]) / world->MaxDistanceToGoal;
}

/*
Normalizes all inputs and then zeroes their mean, such that all inputs range [-1.0,1.0].

This is experimental.
*/
void Q2Agent::_zeroMeanStateVector(const World* world, vector<double>& state)
{
	//note cosine attribute is not normalized, since cosine is inherently normalized
	//_stateHistory[_t][SA_XVELOCITY] /= MAX_VELOCITY;
	//_stateHistory[_t][SA_YVELOCITY] /= MAX_VELOCITY;
	//some magic algebra here. just normalizes and shifts the distance attribute to lie in [-1.0,1.0]
	state[SA_COLLISION_PROXIMITY] =	2.0 * state[SA_COLLISION_PROXIMITY] / MAX_COLLISION_PROXIMITY_RANGE - 1.0;
	//state[SA_GOAL_DIST] = 2.0 * state[SA_GOAL_DIST] / world->MaxDistanceToGoal - 1.0;
}

/*
Given the outputs of the q-nets have already been driven, stochastically selects the best
action according to the output q-values, such that higher q-values have a higher uniform
probability of selection.

For instance, let the q-values of <LEFT,UP,RIGHT,DOWN> be <-2,-3,-4,-6>. Clearly, LEFT is the
absolute optimal action, but a nice idea is to select it probabilistically according to its 
value wrt the other q-values.


Sample python calculation for the same method:
	>>> l
	[-81.2, -81.3, -82, -84]
	>>> shifted = [v - max(l) for v in l]
	>>> shifted
	[0.0, -0.09999999999999432, -0.7999999999999972, -2.799999999999997]
	>>> el = [math.exp(v) for v in shifted]
	>>> el
	[1.0, 0.9048374180359647, 0.4493289641172229, 0.06081006262521814]
	>>> elNormed = [v / sum(el) for v in el]
	>>> elNormed
	[0.41408271379299455, 0.3746775336017786, 0.18605935684745473, 0.025180395757772277]
*/
Action Q2Agent::_getStochasticOptimalAction()
{
	//return _selectionMethod1();
	return _selectionMethod2();
}

//This method uses a more uniform probabilistic selection than method 1,
//which is heavily biased by the most action due to exp().
//TODO: This is incredibly slow for what it accomplishes
Action Q2Agent::_selectionMethod2()
{
	int i;
	double minQval, r, cdf;
	Action selectedAction;
	vector<double> temp;

	temp.resize(NUM_ACTIONS);

	//copy the q-net vals and get the min element
	for(i = 0, minQval = 100000; i < NUM_ACTIONS; i++){
		temp[i] = _currentActionValues[i];
		if(temp[i] < minQval){
			minQval = temp[i];
		}
	}

	//shift the vals by the min (note the min is the most-negative value, hence all values become positive here)
	for(i = 0; i < temp.size(); i++){
		temp[i] -= minQval;
	}

	//The following is for smoothing the cdf, such that the least likely action has a non-zero probability.
	//Since the values were shifted, the lowest-value is now zero; if normalized, its prob will always be zero,
	//hence some ad hoc smoothing helps give it at least some small chance of being selected.
	//get the new non-zero min (the second smallest item in set)
	for(i = 0, minQval = 100000; i < temp.size(); i++){
		if(temp[i] > 0.000001 && temp[i] < minQval){
			minQval = temp[i];
		}
	}
	//add 1/2 of second smallest to all vals to smooth them
	minQval /= 2.0;
	for(i = 0; i < temp.size(); i++){
		temp[i] += minQval;
	}
	//normalize the vals to make them probabilities
	_normalizeVector(temp);

	/*
	cout << "normalized q vals: ";
	for(i = 0; i < temp.size(); i++){
		cout << temp[i] << " ";
	}
	cout << endl;
	*/

	//stochastically choose an action, based on the probabilistic interpretation of the temp vals
	r = (double)(rand() % 1000) / 1000.0;
	//cout << "r: " << r << endl;
	cdf = 0.0;
	selectedAction = ACTION_DOWN;
	for(i = 0; i < temp.size(); i++){ //this chooses from the 'band' of probability mass of each action's qval
		cdf += temp[i];
		if(r < cdf){
			selectedAction = (Action)i;
			break;
		}
	}

	//cout << "selected action: " << GetActionStr((int)selectedAction) << endl;

	return selectedAction;
}

/*
Action Q2Agent::_selectionMethod1()
{
	int i;
	double maxQval, r, cdf;
	Action selectedAction;
	vector<double> temp;

	temp.resize(_qNets.size());

	//copy the q-net vals and get the max element
	for(i = 0, maxQval = -10000000; i < _qNets.size(); i++){
		temp[i] = _qNets[i].GetOutputs()[0].Output;
		if(temp[i] > maxQval){
			maxQval = temp[i];
		}
	}

	//shift the vals down by the max and set them to exp(val)
	for(i = 0; i < temp.size(); i++){
		temp[i] = exp(temp[i] - maxQval);
	}

	//normalize the vals to make them probabilities
	_normalizeVector(temp);

	
	//stochastically choose an action, based on the probabilistic interpretation of the temp vals
	r = (double)(rand() % 1000) / 1000.0;
	cdf = temp[0];
	selectedAction = ACTION_DOWN;
	for(i = 0; i < temp.size(); i++){ //this chooses from the 'band' of probability mass of each action's qval
		if(r < cdf){
			selectedAction = (Action)i;
			break;
		}
		cdf += temp[i];
	}

	return selectedAction;
}
*/

//Normalizes a vector of values, consequently making them probabilities, such that their sum is 1.0.
//This is not algebraic unitization.
void Q2Agent::_normalizeVector(vector<double>& vec)
{
	double vsum = 0.0;

	for(int i = 0; i < vec.size(); i++){
		vsum += vec[i];
	}

	for(int i = 0; i < vec.size(); i++){
		vec[i] /= vsum;
	}
}

/*
How the agent calculates its current state vector, given the world and the state of all missiles.
The definition of the state vector is an incredibly important part of this system.

Current state attrbutes are defined in the .hpp: tyepdef enum StateAttribute{XPOS, YPOS, XVELOCITY, YVELOCITY, OBSTACLE_DIST, MISSILE_LIKELIHOOD, STRIKE_LIKELIHOOD};
*/
void Q2Agent::_deriveActionStates(const World* world, const vector<Missile>& missiles, vector<vector<double>>& actionStates)
{
	//tell the agent if it is headed in a good direction: vector-cosine gives a nice value in range [+1.0,-1.0]
	double xHeading, yHeading, destX, destY;

	//set all the action-unique state values (the state estimate, given the action) for current time _t
	for(int action = 0; action < actionStates.size(); action++){
		//get the velocity/heading state-specific values
		switch(action){
			case ACTION_UP:
				xHeading = 0;
				yHeading = 1.0;
				break;
			case ACTION_RIGHT:
				xHeading = 1.0;
				yHeading = 0;
				break;
			case ACTION_DOWN:
				xHeading = 0;
				yHeading = -1.0;
				break;
			case ACTION_LEFT:
				xHeading = -1.0;
				yHeading = 0;
				break;
			default:
				cout << "ERROR action not found in _updateState()" << endl;
				break;				
		}
		
		//estimate of where the agent will be, given this action
		destX = agent.x + xHeading;
		destY = agent.y + yHeading;
		//only allow checkin valid positions; not those located w/in an obstacle or off-grid
		if(!world->IsValidPosition(destX, destY) || world->GetCell(destX, destY).isObstacle){
			destX = agent.x;
			destY = agent.y;
		}

		//given the info derived from estimating the result of this action (heading, dest location, etc), estimate the state values once we're there
		_estimateSubsequentState(xHeading, yHeading, destX, destY, world, missiles, actionStates[action]);
	}

	//set the distance to the nearest obstacle, given current direction	REMOVED TO _udateState()
	//state[SA_COLLISION_PROXIMITY] = _nearestObjectOnHeading(agent.xVelocity, agent.yVelocity, world, missiles);
	//state[SA_COLLISION_PROXIMITY] = log(1.0 + _nearestObjectOnHeading(agent.xVelocity, agent.yVelocity, world, missiles));

	//get the imminent-missile-hit in current position
	//state[SA_MISSILE_LIKELIHOOD] = _getMissileLikelihood(missiles);

	//get the likelihood of hitting an enemy if we shoot from current position
	//state[SA_STRIKE_LIKELIHOOD] = _getStrikeLikelihood(missiles);

	//get the distance to goal
	//state[SA_GOAL_DIST] = _dist(world->GOAL_X, world->GOAL_Y, agent.x, agent.y);
	//state[SA_GOAL_DIST] = _dist(world->GOAL_X, world->GOAL_Y, agent.x, agent.y);
	//state[SA_GOAL_DIST] = log(1.0 + _dist(world->GOAL_X, world->GOAL_Y, agent.x, agent.y));
}

/*
The state-vector estimator, given all the info about where the agent thinks it will be, given some action, and given
the external environment (world, missiles).
*/
void Q2Agent::_estimateSubsequentState(double xHeading, double yHeading, double destX, double destY, const World* world, const vector<Missile>& missiles, vector<double>& subsequentState)
{
	double x_temp, y_temp;

	//check if this expected location is visited (invalid positions are treated as visited)
	subsequentState[SA_LOCATION_VISIT_COUNT] = 0.0;
	if(!world->IsValidPosition((int)destX, (int)destY)){ //if expected position is invalid, mark it as visited; this is just an undefined edge-case
		subsequentState[SA_LOCATION_VISIT_COUNT] = -2.0;
	}
	else if(world->GetCell((int)destX, (int)destY).traversalCount > 0){
		//estimate up to 20 retraversals
		subsequentState[SA_LOCATION_VISIT_COUNT] = max(-2.0, (double)world->GetCell((int)destX, (int)destY).traversalCount / -10.0);
		/* the old limited-knowledge version
		//check if intended location is in the visited set
		for(int i = 0; i < _recentLocations.size(); i++){
			if(destX == _recentLocations[i].first && destY == _recentLocations[i].second){
				subsequentState[SA_LOCATION_VISIT_COUNT] = -2.0;
			}
		}
		*/
	}
	
	//set the velocity values
	//subsequentState[SA_XVELOCITY] = xHeading;
	//subsequentState[SA_YVELOCITY] = yHeading;
	//set distance per the position delta if this action is taken
	//subsequentState[SA_GOAL_DIST] = _dist(world->GOAL_X, world->GOAL_Y, agent.x + xHeading, agent.y + yHeading);

	//set the distance to nearest object state attribute per the heading given by each ACTION
	subsequentState[SA_COLLISION_PROXIMITY] = _nearestObjectOnHeading(xHeading, yHeading, destX, destY, world, missiles);

	//set the goal-cossim value for each action net
	//IMPORTANT: If agent is on the goal, zero vector is passed to cossim, which is undefined. In this case,
	//clamp cosine to 1.0 (positive/good) for learning consistency with the goal reward.
	x_temp = world->GOAL_X - destX;
	y_temp = world->GOAL_Y - destY;
	if(x_temp == 0 && y_temp == 0){ //check to avert passing zero vector to cossim when agent is directly on the goal
		subsequentState[SA_GOAL_COSINE] = 1.0;
	}
	else{
		subsequentState[SA_GOAL_COSINE] = _cosSim(x_temp, y_temp, xHeading, yHeading);
	}

	/*
	//determine if agent is headed in a direction where it has already been, in the last k steps
	//TODO: try both EMA and centroid/avg
	//destX = _locationEma.first - agent.x;
	//destY = _locationEma.second - agent.y;
	x_temp = _locationAvg.first  - destX;
	y_temp = _locationAvg.second - destY;
	if(x_temp == 0 && y_temp == 0){ //check to avert passing zero vector to cossim when agent is at its previous location estimate
		subsequentState[SA_RECENT_LOCATION_COSINE] = 1.0;
	}
	else{
		//measures cos-sim between the heading vector and the vector pointing in the direction of some estimated of the previous location
		subsequentState[SA_RECENT_LOCATION_COSINE] = _cosSim(x_temp, y_temp, xHeading, yHeading);
	}
	*/
}


/*
The solution here is just for the agent to simulate its current velocity for several time steps to estimate
all the locations it will need to traverse.

TODO: a more efficient way to do this. Could query all points up to a distance of d, such that their cosine sim to current heading
is in some range.

@headingX: x component of some heading
@headingY: y component of some heading
@baseX: Base x-position for comparison. <<< BE SAVVY WITH THESE. DO YOU WANT BASE=Agent.current-position or BASE=agent.hypothetical-position?
@baseY: Base y-position for comparison
Rest are self-explanatory. Distances are relative to agent's current position.
*/
double Q2Agent::_nearestObjectOnHeading(double headingX, double headingY, double baseX, double baseY, const World* world, const vector<Missile>& missiles)
{
	bool collision;
	int x_t, y_t, t;
	double dist = MAX_COLLISION_PROXIMITY_RANGE;

	//simulate next t time steps at current velocities/heading
	for(t = 0, collision = false; t < 15; t++){
		x_t = (int)(baseX + headingX * (double)t);
		y_t = (int)(baseY + headingY * (double)t);
		//check if position is either an obstacle of off-map, which is the same as an impenetrable obstacle
		if(!world->IsValidPosition(x_t, y_t) || world->GetCell(x_t, y_t).isObstacle){
			dist = _dist(baseX, baseY, (double)x_t, (double)y_t) - 1.0; //minus one, since collisions occur for adjacent cells, not when the agent is co-located with some obstacle
			//return the min of the dist or max sensitivity range
			if(dist > MAX_COLLISION_PROXIMITY_RANGE){
				dist = MAX_COLLISION_PROXIMITY_RANGE;
			}
			collision = true; //exit loop for first obstacle/out-of-bounds found since its the min dist
		}
	}

	return dist;
}

double Q2Agent::_dist(double x1, double y1, double x2, double y2)
{
	//points are the same, return zero and avert call to sqrt(0)
	if(x1 == x2 && y1 == y2){
		return 0.0;
	}

	return sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) );
}

/*
Experimental: let's the agent make continous actions (velocities) based on the q values.

void Q2Agent::_takeContinuousAction()
{
	double ycomp = _qNets[(int)ACTION_UP].GetOutputs()[0].Output - _qNets[(int)ACTION_DOWN].GetOutputs()[0].Output;
	double xcomp = _qNets[(int)ACTION_RIGHT].GetOutputs()[0].Output - _qNets[(int)ACTION_LEFT].GetOutputs()[0].Output;
	double ss = xcomp*xcomp + ycomp*ycomp;
	double normal = 0;
	if(ss > 0){
		//the above check averts a call to sqrt(0)
		normal = sqrt( xcomp*xcomp + ycomp*ycomp );
	}

	normalize all the q values
	//for(int i = 0; i < _qNets.size(); i++){
	//	normal += (_qNets[i].GetOutputs()[0].Output * _qNets[i].GetOutputs()[0].Output);
	//}
	//normal = sqrt(normal);
	

	agent.xVelocity = 2 * xcomp / normal;
	agent.yVelocity = 2 * ycomp / normal;
}
*/

/*
Defines the attribute, what is the likelihood of hitting an enemy if we fire from
current position?
*/
double Q2Agent::_getStrikeLikelihood(const vector<Missile>& missiles)
{
	double likelihood = 0.0;

	for(int i = 0; i < missiles.size(); i++){
		if(missiles[i].x == agent.x){ //given agent's current position and orientation (just take guns as always x-facing, for now)
			likelihood = 1.0; // ((double)missiles[i].x - (double)agent.x + 1.0);
		}
	}

	return likelihood;
}

/*
Returns likelihood of a missile hit on the agent, in the current position

Using likelihoods seemed clever, as long as their definition matches reality.
LIkelihoods soften the decision boundary, presumably letting the agent veer from
bad spots, even if they aren't so bad right now.

The true way to check for impending collisions is through solving linear systems of velocity
and current position of agent/missile to detect if their is a solution. That's expensive,
so heuristics are used instead. We especially want to detect very near (in time/space) collisions,
not those far out in time/distance.

*/
double Q2Agent::_getMissileLikelihood(const vector<Missile>& missiles)
{
	double likelihood = 0.0;
	
	for(int i = 0; i < missiles.size(); i++){
		/*potential 3-timestep algorithm to detect impending missile strike likelihood
		for 3-5 time steps, check if there is a linear solution in pos/vel to missile/agent.
		If so, or if distance is very small, then a collision is imminent. Could weight 
		these by time/distance, such that events further out in time/space have lower importance.
		*/

		//assume the missiles travel straight down; if we're in the same column, the likelihood is weighted by distance
		if(missiles[i].x == agent.x){
			likelihood = 1.0 / (min(_dist(missiles[i].x, missiles[i].y, agent.x, agent.y), 1.0) + 1.0);
		}
		
		if(missiles[i].x == agent.x - 1){
			likelihood = 0.5 / (min(_dist(missiles[i].x, missiles[i].y, agent.x, agent.y), 1.0) + 1.0);
		}
		if(missiles[i].x == agent.x + 1){
			likelihood = 0.5 / (min(_dist(missiles[i].x, missiles[i].y, agent.x, agent.y), 1.0) + 1.0);
		}
	}

	return likelihood;
}


/*A flat distance to nearest object, w/out respect to direction, just gives preference
//for the agent to stay away from obstacles.
double Q2Agent::_nearestObstacleDist(const World* world)
{
	double d, dist = 10.0;

	//check within a radius of 10 cells for nearest obstacle
	for(int i = max((int)agent.x - 5, 0); i < min((int)world->MaxX(), (int)agent.x + 5); i++){
		for(int j = max((int)agent.y - 5, 0); j < min((int)world->MaxY(), (int)agent.y + 5); j++){
			if(world->GetCell(i,j).isObstacle){
				d = _dist((double)i,(double)j,agent.x,agent.y);
				if(d < dist){
					dist = d;
				}
			}
		}
	}

	return dist;
}
*/

//copies a vector to another; user beware on size checking
void Q2Agent::_copyVec(vector<double>& v1, vector<double>& v2)
{
	for(int i = 0; i < v1.size(); i++){
		v2[i] = v1[i];
	}
}

//Based on the agent's last action, determines if action result in a wall/boundary collision.
bool Q2Agent::_isWallCollision(const World* world)
{
	if(agent.x == 0 && CurrentAction == ACTION_LEFT)
		return true;
	if(agent.y == 0 && CurrentAction == ACTION_DOWN)
		return true;
	if(agent.x == world->MaxX() && CurrentAction == ACTION_RIGHT)
		return true;
	if(agent.y == world->MaxY() && CurrentAction == ACTION_UP)
		return true;

	return false;
}

/*
This is the classical, discrete q-learning reward function, for which the agent typically only
receives a reward (positive or negative) when it reaches some sort of terminal state. This is very likely
to diverge for the approximate q-learning model, but needed to be experimented with anyway. The reason
a discrete/terminal reward function works in discrete q-learning is because the values propagate recursively
to other states, but this is not (immediately anyhow) the case with approximate q-learning, for which the 
neural net is going to experience "sudden", oscillating reward values.

Its still worth testing this, since there may be simple mechanisms and strategies of overcoming these effects.

Results: This has been shown to work, but the agent's behavior is imprecise around the goal. It approximates the
goal location well, but spends a lot of time circling. It's learning must also be constrained: I got this to work
by only updating q-values when the reward was non-zero (hence, the agent was only updated when it experienced what
might be interpreted as 'significant' events, collision, goal-reached, etc). I used eta = 0.05, gamma = 0.9,
network with 12 hidden units, otherwise mostly the same as the normal architecture. 
*/
double Q2Agent::_getCurrentRewardValue_Terminal(const World* world, const vector<Missile>& missiles)
{
	double reward = 0.0;

	if(world->GetCell(agent.x, agent.y).isGoal){
		reward += EXTERNAL_REWARD_GOAL;
	}
	else if(agent.sufferedCollision){
		reward += EXTERNAL_REWARD_COLLISION;
	}
	
	if(world->GetCell(agent.x, agent.y).traversalCount > 0){
		//bound negative re-visit rewards by 20 revisits
		if(world->GetCell(agent.x, agent.y).traversalCount >= 20){
			reward += (EXTERNAL_REWARD_VISITED * 20);
		}
		else{
			reward += (EXTERNAL_REWARD_VISITED * (double)world->GetCell(agent.x, agent.y).traversalCount);
		}
	}

	if(reward != 0.0){
		StoreTerminalState(reward);
	}

	return reward;
}

/*
Updates the agent's externally-received reward value. Don't confuse this with the reward coefficient vector,
or the reward() function, which returns the agent's estimate of the current reward.
*/
double Q2Agent::_updateExternalReward(const World* world, const vector<Missile>& missiles)
{
	double reward = 0.0;

	if(world->GetCell(agent.x, agent.y).isGoal){
		reward += EXTERNAL_REWARD_GOAL;
		_kVectors.push_back(kvector(_getCurrentState((Action)CurrentAction), reward, ALPHA_GOAL_REACHED));
	}
	else{
		if(agent.sufferedCollision){
			reward += EXTERNAL_REWARD_COLLISION;
			_kVectors.push_back(kvector(_getCurrentState((Action)CurrentAction), reward, ALPHA_COLLISION));
		}
		
		//for this to be useful the agent needs previous-location estimate data in the xs (state), which it gets through the cosine-visited attribute
		if(world->GetCell(agent.x, agent.y).traversalCount > 0){
			//bound negative re-visit rewards by 20 revisits
			if(world->GetCell(agent.x, agent.y).traversalCount >= 20){
				reward += (EXTERNAL_REWARD_VISITED * 20);
			}
			else{
				reward += (EXTERNAL_REWARD_VISITED * (double)world->GetCell(agent.x, agent.y).traversalCount);
			}
			//reward += EXTERNAL_REWARD_VISITED;
			_kVectors.push_back(kvector(_getCurrentState((Action)CurrentAction), reward, ALPHA_REPETITION));
		}
	}

	if(reward != 0.0){
		StoreTerminalState(reward);
	}

	return reward;
}

/*
Flushes all of the k-vectors to whatever file they're going to.
*/
void Q2Agent::_flushRewardVectors(bool clearVecs)
{
	for(int i = 0; i < _kVectors.size(); i++){
		kvector& kv = _kVectors[i];
		//output all of the xs (the state vector values when reward was received)
		for(int j = 0; j < kv.xs.size(); j++){
			_kVectorFile << kv.xs[j] << ",";
		}
		_kVectorFile << kv.r << "," << kv.alpha << endl;
	}

	if(clearVecs){
		//clear current vectors
		_kVectors.clear();
	}
}


/*
The EM version of the reward function. Here, the alpha-neurons are just driven,
then we calculate the risk across all of them.
*/
double Q2Agent::_getCurrentRewardValue_Logistic(const World* world, const vector<Missile>& missiles)
{
	const double one = 1.0;
	double reward = 0.0, value;

	//drive all the alpha neurons
	for(auto it = _alphaNeurons.begin(); it != _alphaNeurons.end(); ++it){
		Neuron& neuron = it->second;
		//set up the neuron inputs (indices are shifted since the 0th input is the bias
		neuron.Inputs[0] = &one; //the bias
		neuron.Inputs[SA_GOAL_COSINE+1] = &_stateHistory[_t][(int)CurrentAction][SA_GOAL_COSINE];
		neuron.Inputs[SA_RECENT_LOCATION_COSINE+1] = &_stateHistory[_t][(int)CurrentAction][SA_RECENT_LOCATION_COSINE];
		neuron.Inputs[SA_COLLISION_PROXIMITY+1] = &_stateHistory[_t][(int)CurrentAction][SA_COLLISION_PROXIMITY];
		neuron.Inputs[SA_LOCATION_VISIT_COUNT+1] = &_stateHistory[_t][(int)CurrentAction][SA_LOCATION_VISIT_COUNT];
		//and stimulate
		neuron.Stimulate();
		cout << "reward neuron " << it->first << " prob: " << neuron.Output << endl;
		//get value for this component
		switch((int)it->first){
			case 'g':
				value = EXTERNAL_REWARD_GOAL;
				break;
			case 't':
				value = EXTERNAL_REWARD_VISITED;
				//value = 0;
				break;
			case 'c':
				value = EXTERNAL_REWARD_COLLISION;
				break;
			default:
				cout << "ERROR no case found for " << it->first << " in _getCurrentRewardValue_Logistic" << endl;
				break;
		}
		//update the expected value with this component
		reward += (neuron.Output * value);
	}

	cout << "reward: " << reward << endl;
	return reward;
}



/*
Some experimenting with learning the reward function values, which is the real goal of 
reinforcement learning.

NOTE: Make sure parameters correspond with the order of state attributes as they are output to
training files: SA_GOAL_COSINE, SA_RECENT_LOCATION_COSINE, SA_COLLISION_PROXIMITY
*/
double Q2Agent::_getCurrentRewardValue_Manual1(const World* world, const vector<Missile>& missiles)
{
	//int x_prime, y_prime;
	double reward = 0.0;
	//TODO: better to define rewards as positive/negative or all negative? all negative seems more consistent.
	//double MISSILE_DAMAGE_COST = -10.0;
	//double COLLISION_COST = -5.0;
	//double REPETITION_COST = 0.0;
	//double GOAL_REWARD = 5.0;
	//the unknown coefficients; hard-coding is cheating, the point is to learn these
	//double _coefGoalCos, _coefVisitedCos, _coefCollisionProx;


	//opt, with linear regression: 0.16115334 -0.08905619  0.83468276
	//_coefVisitedCos = -1.0; // the coefficient for the similarity of the agent's current location versus its where it has visited
	//_coefGoalCos = 1.0;
	//_coefCollisionProx = 1.0;
	_coefGoalCos = 0.7;
	_coefVisitedCos = -0.845; // the coefficient for the similarity of the agent's current location versus its where it has visited
	_coefCollisionProx = 0.5;

	/*check if last action caused a collision
	if(agent.sufferedCollision){
		cout << "COLLISION ep=" << _episodeCount << endl;
		//reward += COLLISION_COST;
	}
	*/

	/*
	if(world->GetCell(agent.x, agent.y).isGoal){
		cout << "FOUND GOAL!" << endl;
		//reward += GOAL_REWARD;
	}
	else{
		reward += (coef_GoalDist * _stateHistory[_t][(int)CurrentAction][SA_GOAL_DIST]);
	}
	*/

	//update the cosine-based reward
	reward += (_coefGoalCos * _stateHistory[_t][CurrentAction][SA_GOAL_COSINE]);

	//punish for revisiting locations (cossim is negated, since we desire dissimilarity)
	reward += (_coefVisitedCos * _stateHistory[_t][(int)CurrentAction][SA_RECENT_LOCATION_COSINE]);

	//add in a punishment for distance to nearest obstacle on heading (experimental; NOTE this requires state vector collision proximity attribute has been set)
	//reward += (-MAX_COLLISION_PROXIMITY_RANGE + _stateHistory[_t][(int)CurrentAction][SA_COLLISION_PROXIMITY]);
	reward += (_coefCollisionProx * _stateHistory[_t][(int)CurrentAction][SA_COLLISION_PROXIMITY]);

	//cout << "reward: " << reward << endl;
	return reward;
}

double Q2Agent::_getCurrentRewardValue_Manual2(const World* world, const vector<Missile>& missiles)
{
	int x_prime, y_prime;
	double reward = 0.0;
	//TODO: better to define rewards as positive/negative or all negative? all negative seems more consistent.
	//double MISSILE_DAMAGE_COST = -10.0;
	double COLLISION_COST = -20.0;
	//double REPETITION_COST = -2.0;
	double GOAL_REWARD = 20.0;
	//double RADAR_RADIUS = 8.0;

	/*
	//HARD PARAMETERS
	//Found goal, so return GOAL_REWARD no matter the preceding state or other reward parameters
	if(world->GetCell(agent.x, agent.y).isGoal){
		cout << "FOUND GOAL!" << endl;
		return GOAL_REWARD;
	}
	*/

	//check if last action caused a collision
	if(agent.sufferedCollision){
		cout << "COLLISION ep=" << _episodeCount << endl;
		reward += COLLISION_COST;
	}

	if(world->GetCell(agent.x, agent.y).isGoal){
		cout << "FOUND GOAL!" << endl;
		reward += GOAL_REWARD;
	}
	else{
		//goal distance scaled by 10.0/max-distance provides value of distance cost; this gives a cost in range [0,-10]
		reward += (_dist(world->GOAL_X, world->GOAL_Y, agent.x, agent.y) * (-10.0 / world->MaxDistanceToGoal));
	}

	//add in a punishment for distance to nearest obstacle on heading (experimental; NOTE this requires state vector collision proximity attribute has been set)
	reward += (-MAX_COLLISION_PROXIMITY_RANGE + _stateHistory[_t][(int)CurrentAction][SA_COLLISION_PROXIMITY]);

	//nudge agent in the direction of the goal; buried some linear algebra here. Subtracting these vectors gives the direction vector of the goal relative to the agent
	x_prime = world->GOAL_X - agent.x;
	y_prime = world->GOAL_Y - agent.y;
	//dot product of the velocity vector and pointwise goal direction vector gives value in desirable range -1.0 to 1.0;
	//this value is shifted down by 1.0 to be consistent with the negative-rewards schema
	//note this is only a measure of direction, not magnitude
	//reward += 5 * (_normalizedCosSim(x_prime, y_prime, agent.xVelocity, agent.yVelocity) - 1.0);
	reward += (5.0 * (_normalizedCosSim(x_prime, y_prime, agent.xVelocity, agent.yVelocity) - 1.0));
	//reward += 5 * _cosSim(x_prime, y_prime, agent.xVelocity, agent.yVelocity);
	//cout << "cossim: " << _normalizedCosSim(x_prime, y_prime, agent.xVelocity, agent.yVelocity) << endl;

	/*check if current location was already visited (favor new areas)
	if(world->GetCell(agent.x, agent.y).isTraversed){
		reward += REPETITION_COST;
	}
	*/

	//TODO: This could incorporate distance-weighted trigonometric/trajectory data: use cosine-metric to assess basic co-linearity of agent and missiles
	//distance-weighted sum of missiles within some radius of the agent, essentially its 'radar' signal
	//think of two missiles at the edge of some x radius circle centered at the agent, subtended by angle theta; let the negativity
	//of this situation be defined by a single missile twice the size located directly between the two, at theta/2 (a kind of missile centroid)
	/*for(int i = 0; i < missiles.size(); i++){
		double d = _dist(missiles[i].x, missiles[i].y, agent.x, agent.y) + 1.0; //plus one so we don't have a zero denom below
		if(d < RADAR_RADIUS){ //missiles in range, so penalize it by its distance
			reward += -(1.0 / (d + 1.0));
		}
	}
	*/

	//normalize the reward (this must correspond with neural net architecture for estimating R
	//reward = (reward + 10.0) / 20.0; //normalizes reward to lie approximately in [-1.0,1.0] for TANH neural net output

	//cout << "reward: " << reward << endl;
	return reward;
}


//just normalizes the input vectors before calculating their cosine similarity
double Q2Agent::_normalizedCosSim(double x1, double y1, double x2, double y2)
{
	double norm1 = sqrt(x1*x1 + y1*y1);
	double norm2 = sqrt(x2*x2 + y2*y2);

	if(norm1 == 0 || norm2 == 0){
		return 0.0;
	}

	return _cosSim(x1/norm1, y1/norm1, x2/norm2, y2/norm2);
}

//cosine similarity for 2d vectors; range neatly fits between -1 and 1.0
double Q2Agent::_cosSim(double x1, double y1, double x2, double y2)
{
	double sim;
	double dp = x1*x2 + y1*y2;
	double denom = sqrt(x1*x1 + y1*y1) * sqrt(x2*x2 + y2*y2);
	
	//cout << "args: " << x1 << " " << y1 << " " << x2 << " " << y2 << endl;

	//TODO: This is undefined. What should be returned if a zero-vector is passed? Are there other entry points for the result in the conditional?
	//avert div-zero if either sum-of-squares is zero.
	if(denom == 0.0 || dp == 0.0){
		sim = 0.0;
	}
	else{
		sim = dp / denom;
	}
	//cout << "cossim: " << cossim << endl;

	return sim;
}

/*
TODO: define this function, theoretically. This is where I have hidden the agent's
realization of the last Q-value, given the agent's last action.

For now, I've defined the following reward function, incorporating these parameters:
	hard parameters:
		-was the last action's realization a bonk into an obstacle
		-did the agent just get struck by a missile (most negative parameter)
	softer parameters:
		-whether the current column is a missile trajectory
		-sum-weighted distance to all missiles w/in some radius (eg, within five cells)

Note for now I'm letting the agent's action be transparent wrt reality: the agent can travel
into obstacles, it just realizes a negative value when it does so. Likewise when it gets struck
by a missile. This avoids the game-programming aspect of things for now, like enforcing different
outcomes. In fact, such transparent treatment is kind of clever, even for a game.

I have not defined these parameter settings in a closed form, I've just chosen them manually.


But note that they ought to be made explicit, since they will likely define the 

double Q2Agent::_getTargetQValue(const vector<vector<WorldCell> >& world, const vector<Missile>& missiles)
{
	double qTarget = 0.0;

	qTarget = _getCurrentRewardValue_Manual(world, missiles) + _gamma * _getMaxQValue();

	return ;
}
*/


/*
A small historian for recording example inputs, target values, estimated q value, and action.

Q-Learning uses online, *non-stochastic* neural network updating; this purpose of recording 
examples is for offlining learning for more stable network convergence.

TODO: This could be put into a more flexible Historian class, which is a bit class eager. But for experiments,
I likely want a more flexible way of sampling examples, etc. For instance, storing examples with the greatest error,
etc.

Prereqs: Output stream is open.
*/
void Q2Agent::_recordExample(const vector<double>& state, double qTarget, double qEstimate, Action action)
{
	//write all the state values
	_outputFile << "<";
	for(int i = 0; i < state.size(); i++){
		if(i < state.size() - 1){
			_outputFile << state[i] << ",";
		}
		else{
			_outputFile << state[i] << ">";
		}
	}
	//output all the remaining values
	_outputFile << "," << qTarget << "," << qEstimate << "," << GetActionStr(CurrentAction) << endl;
}

//TODO: this is pretty disorganized, esp wrt the World
void Q2Agent::ResetEpoch()
{
	//store performance given the last epoch
	_lastEpochCollisionRate = EpochCollisionCount / (double)_episodeCount;
	_lastEpochActionCount = _episodeCount;
	_lastEpochReward = _epochReward;

	_epochCount++;
	_episodeCount = 0;
	_epochReward = 0;

	EpochGoalCount = 0;
	EpochActionCount = 0;
	EpochCollisionCount = 0;

	//reset location history
	_locationAvg.first = _locationAvg.second = 0.0;
	_locationAvg.first = _locationEma.second = 0.0;
	for(int i = 0; i < _recentLocations.size(); i++){
		_recentLocations[i].first = _recentLocations[i].second = 0.0;
	}
	
	//_storeTerminalState(_getCurrentState((Action)CurrentAction),terminalReward);
}

//Just a wrapper. Let's the agent determine its terminal state, but client passes in terminal value.
void Q2Agent::StoreTerminalState(double terminalReward)
{
	_storeTerminalState(_getCurrentState((Action)CurrentAction),terminalReward);
}

/*
Can be used to store the snapshot of the current reward parameters, and some measure of the reward experienced
by those parameters.
*/
void Q2Agent::StoreRewardParams(const vector<double>& rewardParams, double reward)
{
	_storeLabeledVector(rewardParams, reward, _rewardParamsFile);
}

/*
For testing/experimentation: log the terminal state vectors.

The motivation is to use the terminal states as prototypes of +/-1
labelled vectors, as a basis for learning the reward function itself.

For instance, agent crashes when state vector is x1,x2,x3 log this as x1,x2,x3,-1.
this way the agent could periodically (after logging many epochs) learn the terminal-prototypes
to determine the reward function parameters.


@state: A vector of state data which was held when the agent reached some terminal condition (a snapshot).
@terminalValue: Some value, which for now ought to be just +/-1. The values could be a scalar, but its
subject to experiment whether or not that will help at all.
*/
void Q2Agent::_storeTerminalState(const vector<double>& state, double terminalValue)
{
	_storeLabeledVector(state, terminalValue, _prototypeFile);
}

//util for storing an example in csv, such as for offline training
void Q2Agent::_storeLabeledVector(const vector<double>& state, double terminalValue, fstream& outputFile)
{
	for(int i = 0; i < state.size(); i++){
		outputFile << state[i] << ",";
	}
	outputFile << terminalValue << endl;
}

/*

*/
void Q2Agent::LoopedUpdate(const World* world, const vector<Missile>& missiles)
{
	bool convergence = false;
	int action, iterations;
	double netError = 0.0, maxQ = 0.0, tempMax = 0.0, lastMax = 0.0, qTarget = 0.0, reward = 0.0;
	vector<double> tempQVals;
	Action lastOptimalAction = ACTION_DOWN, optimalAction = ACTION_LEFT, tempMaxAction = ACTION_RIGHT;

	//Update agent's current state and state history
	_updateCurrentActionStates(world, missiles);

	//get the target q factor from the experienced reward given the last action
	//reward = _getCurrentRewardValue_Manual(world, missiles);
	reward = _getCurrentRewardValue_Logistic(world, missiles);
	cout << "reward: " << reward << endl;

	//loop over the q values, retraining to result in some degree of convergence, at least for this action
	maxQ = -999999999;
	lastMax = -999999999;
	tempQVals.resize(_currentActionValues.size());
	for(iterations = 0, convergence = false; iterations < 100 && !convergence; iterations++){
		//classify the new current-state across all actions
		for(action = 0, tempMax = -1000000; action < NUM_ACTIONS; action++){
			//classify the state we just entered, given the previous action
			_qNet.Classify(_getCurrentState((Action)action));
			tempQVals[action] = _qNet.GetOutputs()[0].Output;
			_currentActionValues[action] = _qNet.GetOutputs()[0].Output;
			//track the max action available in current state
			if(tempQVals[action] > tempMax){
				tempMax = tempQVals[action];
				tempMaxAction = (Action)action;
			}
		}
		lastMax = maxQ;
		lastOptimalAction = optimalAction;
		optimalAction = tempMaxAction;
		maxQ = tempMax;

		//detect convergence: if estimate is within 0.1 of previous estimate (verifying also that this is consistently the same action)
		netError = _absDiff(lastMax,maxQ);
		cout << "maxq " << maxQ << "  lastMax " << lastMax << "  netError " << netError << endl;
		convergence = (lastOptimalAction == optimalAction) && (netError < 0.05);
		if(!convergence){
			qTarget = reward + _gamma * maxQ;
			cout << "QTARGET: " << qTarget << endl;
			//backpropagate the error and update the network weights for the last action (only)
			_qNet.Classify(_getPreviousState((Action)CurrentAction)); //the net must be re-clamped to the previous state inputs and signals
			_qNet.BackpropagateError(_getPreviousState((Action)CurrentAction), qTarget);
			_qNet.UpdateWeights(_getPreviousState((Action)CurrentAction), qTarget);
		}
	}

	_epochReward += qTarget;

	//record this example
	if(_episodeCount > 100){
		_recordExample(_getPreviousState((Action)CurrentAction), qTarget, _currentActionValues[CurrentAction], CurrentAction);
	}

	//take the action with the current highest q-value
	CurrentAction = optimalAction;

	//e-greedy action selection: select the optimal action 80% of the time
	//if(rand() % (1 + (_episodeCount / 2000)) == (_episodeCount / 2000)){ //diminishing stochastic exploration
	if((rand() % 5) == 4){
	//if(((rand() % 5) == 4 && _totalEpisodes < 1000) || ((_totalEpisodes >= 1000) && (rand() % 10) == 9)){
		if(rand() % 2 == 0)
			CurrentAction = _getStochasticOptimalAction();
		else
			CurrentAction = (Action)(rand() % NUM_ACTIONS);
	}

	//map the action into outputs
	_takeAction(CurrentAction);

	//some metalogic stuff: random restarts and force agent to move if in same place too long
	//TODO: this member represents bad coupling
	agent.sufferedCollision = false;
	_episodeCount++;
	_totalEpisodes++;

	//testing: print the neural net weights
	//if(_episodeCount % 100 == 1){
		_qNet.PrintWeights();
	//}
}

//Given a line, tokenize it using delim, storing the tokens in output
void Q2Agent::_tokenize(const string &s, char delim, vector<string> &tokens)
{
    stringstream ss(s);
    string temp;
	
	//clear any existing tokens
	tokens.clear();
	
    while (getline(ss, temp, delim)) {
        tokens.push_back(temp);
    }
}

/*

*/
void Q2Agent::DirectApproximationUpdate(const World* world, const vector<Missile>& missiles)
{
	int action = 0;
	double maxQ = 0, qTarget = 0, rewardTarget = 0, rewardEstimate = 0, prevEstimate = 0;
	Action optimalAction = ACTION_UP;

	_qNet.SetEta(0.02);
	//TODO: momentum is good in general, but I'm not sure the effect in this context. In general it speeds training and helps escape local minima.
	_qNet.SetMomentum(0.0);
	//set the regularization term
	_qNet.SetWeightDecay(0.0001);

	//Update agent's current state and state history for all possible actions
	_updateCurrentActionStates(world, missiles);

	//Update external rewards (agent ran into wall, reached goal, etc)
	_rewardApproximator.SetEta(0.05);
	_rewardApproximator.SetMomentum(0.0);
	rewardTarget = _updateExternalReward(world,missiles);
	_rewardApproximator.Classify(_getCurrentState((Action)action));
	rewardEstimate = _rewardApproximator.GetOutputs()[0].Output;
	//only backpropagate non-zero external rewards (significant events)
	if(rewardTarget != 0.0 && _totalEpisodes < 10000){
		for(int i = 0; i < 4; i++){
		cout << "updating reward net" << endl;
		_rewardApproximator.BackpropagateError(_getCurrentState((Action)action), rewardTarget);
		_rewardApproximator.UpdateWeights(_getCurrentState((Action)action), rewardTarget);
		_rewardApproximator.Classify(_getCurrentState((Action)action));
		rewardEstimate = _rewardApproximator.GetOutputs()[0].Output;
		}
	}

	for(int i = 0; i < 1; i++){
	//classify the new current-state across all action-nets 
	for(action = 0, maxQ = -10000000; action < NUM_ACTIONS; action++){
		//classify the state we just entered, given the previous action
		_qNet.Classify(_getCurrentState((Action)action));
		//cout << GetActionStr(action) << "\t" << _qNet.GetOutputs()[0].Output << endl;
		_currentActionValues[action] = _qNet.GetOutputs()[0].Output;
		//track the max action available in current state
		if(_qNet.GetOutputs()[0].Output > maxQ){
			maxQ = _qNet.GetOutputs()[0].Output;
			optimalAction = (Action)action;
		}
	}
	
	//get the target q factor from the experienced reward given the last action
	//double reward = _getCurrentRewardValue_Manual1(world, missiles);
	//double reward = _getCurrentRewardValue_Logistic(world, missiles);
	cout << "reward: " << rewardEstimate << " actual: " << rewardTarget << endl;
	qTarget = rewardEstimate + _gamma * maxQ;
	//cout << "QTARGET: " << qTarget << endl;
	_epochReward += qTarget;
	//cout << "qTarget: " << qTarget << " maxQ: " << maxQ << endl;

	//this seems to work nicely, to prevent unlearning from many non-terminal experiences
	//if(rewardEstimate != 0){
	if(_totalEpisodes < 20000){
		//cout << "currentaction " << (int)CurrentAction << " qnets.size()=" << _qNets.size() << endl;
		//backpropagate the error and update the network weights for the last action (only)
		const vector<double>& previousState = _getPreviousState((Action)CurrentAction);
		_qNet.Classify(previousState); //the net must be re-clamped to the previous state's signals
		prevEstimate = _qNet.GetOutputs()[0].Output;
		//cout << "prev estimate: " << prevEstimate << endl;
		_qNet.BackpropagateError(previousState, qTarget);
		_qNet.UpdateWeights(previousState, qTarget);
		//cout << "44" << endl;
	}
	}

	if(_totalEpisodes > 100){
		//record this example; this is useful for both replay-based learning and for data analysis
		_recordExample(_getPreviousState((Action)CurrentAction), qTarget, prevEstimate, CurrentAction);
		//record the labeled external rewards as well, every so often
		//if(_totalEpisodes % 1000 == 0){
		//	_flushRewardVectors();
		//}
	}

	//take the action with the highest q-value
	//LastAction = CurrentAction;
	CurrentAction = optimalAction;

	//randomize the action n% of the time
	//if(rand() % (1 + (_episodeCount / 2000)) == (_episodeCount / 2000)){ //diminishing stochastic exploration
	if((rand() % 5) == 4 && _totalEpisodes < 20000){
		if(rand() % 2 == 0)
			CurrentAction = _getStochasticOptimalAction();
		else
			CurrentAction = (Action)(rand() % NUM_ACTIONS);
	}
	//experimental: search for optimal action
	else{
		_searchForOptimalAction(world, missiles, 3);
	}

	//map the action into outputs
	_takeAction(CurrentAction);

	//some metalogic stuff: random restarts and force agent to move if in same place too long
	//TODO: this member represents bad coupling
	agent.sufferedCollision = false;
	_episodeCount++;
	_totalEpisodes++;

	//testing: print the neural net weights
	//if(_episodeCount % 100 == 1){
		//_qNet.PrintWeights();
	//}
}


/*
Learning experiment for which the agent experiences external rewards (collisions, goals, etc),
periodically updating its reward() function based on these external rewards. These reward()
function parameters then drive the q-value net.

Here, the reward approximation works as follows:
	1) Take actions in the world, giving a bag of vectors <states, external reward, alpha>, where 

TODO: Test using both the multinomial and 'ovr' (one-versus-rest) built in schemes in python's/sklearn's logistic-regression.

RESULTS: This sometimes converges to optimal, zero-collision behavior. However I still cant justify how/if the result is correct!
The agent's behavior is often subject to traps, but in many cases converges to perfect, trap/collision free, behavior! WHY? The result
may be illegitimate; I suspect the q-net is doing the work. If it does work, this may be a case for 'pocket', of re-running
logistic behavior for difference experience-sets, and snapshotting the alpha-net parameters for which the best performance
was obtained.
*/
void Q2Agent::LogisticRewardApproximationUpdate(const World* world, const vector<Missile>& missiles)
{
	_qNet.SetEta(0.01);
	if(_totalEpisodes > 1000){
		_qNet.SetEta(0.05);
	}
	if(_totalEpisodes > 5000){
		_qNet.SetEta(0.1);
	}
	//Theoretically momentum may be bad for online learning: since the inputs are so correlated, momentum causes unlearning (too much local adaptation).
	_qNet.SetMomentum(0.0);
	_qNet.SetWeightDecay(0.001); //Arbitrary. Larger (~0.01) seems to indeed improve results, but be wary of it encouraging traps.
	_gamma = 0.9;

	if(STATE_DIMENSION != 4){
		//changes to state dimension affect the parsing below; just a note to self
		cout << "ERROR: update cases in LogisticRewardApproximationUpdate() to reflect dim+1" << endl;
	}

	//update the reward function parameters every k stimuli, for some large k
	if(_kVectors.size() > 5000){
		string junk, line;
		//for the sake of experimentation, I'm just outputting the k-vectors, mining them in python, then reading the output params back in
		_flushRewardVectors(true);
		cout << "Enter anything to continue, once python logistic regression has completed, and params can be read from rwdParams.csv" << endl;
		cin >> junk;
		//now read the reward params back in to each neuron
		fstream paramFile;
		paramFile.open("rwdParams.csv", ios::in);
		while(getline(paramFile, line)){
			vector<string> toks;
			pair<char, vector<double>> alphaParams;

			//parse this csv line
			_tokenize(line, ',', toks);
			for(int i = 0; i < toks.size(); i++){
				//parsing csv tokens: params file is expected to be formatted as alpha,alpha-reward,coef0,coef1,coef2,coef3
				cout << "tok" << i << ": " << toks[i] << endl;
				switch(i){
					case 0:
						alphaParams.first = toks[i][0];
						break;
					case 1: //skip reward value
						break;
					case 2: //parse the coefs
					case 3:
					case 4:
					case 5:
					case 6:
						alphaParams.second.push_back(stod(toks[i]));
						break;
					default:
						cout << "ERROR parse category " << i << " not found in RewardApproximationUpdate of tok " << toks[i] << endl;
						cout << "and line " << line << endl;
						break;
				}
			}
			//load these reward parameters into this alpha's neuron
			Neuron& neuron = _alphaNeurons.at(alphaParams.first);
			if(neuron.Weights.size() != alphaParams.second.size()){
				cout << "ERROR parsing error in RewardApproximationUpdate(). " << neuron.Weights.size() << " != " << alphaParams.second.size() << endl;
			}
			for(int i = 0; i < neuron.Weights.size(); i++){
				cout << "setting weight " << i << " of " << alphaParams.first << " neuron to " << alphaParams.second[i] << endl;
				neuron.Weights[i].w = alphaParams.second[i];
				neuron.Weights[i].dw = 0;
			}
			//string junk3;
			//cout << "enter something" << endl;
			//cin >> junk3;
		}
	}
	
	//run the regular localized update
	Update(world, missiles);
}


/*
Precon: ExperiencedQValue has already been set, given the action the agent just took

-feed state to all neural nets, for each action
-exploitatively take best action 
-feed experienced reward back to neural net

helpful: http://outlace.com/Reinforcement-Learning-Part-3/

TODO: Edge cases can interfere with learning. For instance, when the agent finds the goal, it is teleported to 
a random location. Two problems arise. 1) The state in which it finds the goal can be degenerate, depending on the
state attributes, and those degenerate values are then backpropagated with a large error term, given the large reward
for finding the foal. For instance, at the Goal, the agent's cosine state attribute is undefined (since the position vectors are equal)
and must be clamped at 1.0, instead of feeding back 0.0 (the default return of cossim() for zero vectors). 2) The transition from
the goal state/location to some random location means the agent will try to learn that random transition. Learning must instead
be restarted so the epochs are completely separated.
*/
void Q2Agent::Update(const World* world, const vector<Missile>& missiles)
{
	int action;
	double maxQ, qTarget, prevEstimate;
	Action optimalAction = ACTION_UP;
	
	_qNet.SetEta(0.05);

	//Update agent's current state and state history for all possible actions
	_updateCurrentActionStates(world, missiles);

	//Update external rewards (agent ran into wall, reached goal, etc)
	_updateExternalReward(world, missiles);

	//classify the new current-state across all action-nets 
	for(action = 0, maxQ = -10000000; action < NUM_ACTIONS; action++){
		//classify the state we just entered, given the previous action
		_qNet.Classify(_getCurrentState((Action)action));
		//cout << GetActionStr(action) << "\t" << _qNet.GetOutputs()[0].Output << endl;
		_currentActionValues[action] = _qNet.GetOutputs()[0].Output;
		//track the max action available in current state
		if(_qNet.GetOutputs()[0].Output > maxQ){
			maxQ = _qNet.GetOutputs()[0].Output;
			optimalAction = (Action)action;
		}
	}
	
	//get the target q factor from the experienced reward given the last action
	//double reward = _getCurrentRewardValue_Manual1(world, missiles);
	//double reward = _getCurrentRewardValue_Logistic(world, missiles);
	double reward = -1.0;
	//cout << "reward: " << reward << endl;
	qTarget = reward + _gamma * maxQ;
	//cout << "QTARGET: " << qTarget << endl;
	_epochReward += qTarget;
	//cout << "qTarget: " << qTarget << " maxQ: " << maxQ << endl;

	if(_totalEpisodes < 15000){
		//cout << "currentaction " << (int)CurrentAction << " qnets.size()=" << _qNets.size() << endl;
		//backpropagate the error and update the network weights for the last action (only)
		const vector<double>& previousState = _getPreviousState((Action)CurrentAction);
		_qNet.Classify(previousState); //the net must be re-clamped to the previous state's signals
		prevEstimate = _qNet.GetOutputs()[0].Output;
		//cout << "prev estimate: " << prevEstimate << endl;
		_qNet.BackpropagateError(previousState, qTarget);
		_qNet.UpdateWeights(previousState, qTarget);
		//cout << "44" << endl;
	}

	if(_totalEpisodes > 100){
		//record this example; this is useful for both replay-based learning and for data analysis
		_recordExample(_getPreviousState((Action)CurrentAction), qTarget, prevEstimate, CurrentAction);
		//record the labeled external rewards as well, every so often
		//if(_totalEpisodes % 1000 == 0){
		//	_flushRewardVectors();
		//}
	}

	//take the action with the highest q-value
	//LastAction = CurrentAction;
	CurrentAction = optimalAction;

	//randomize the action n% of the time
	//if(rand() % (1 + (_episodeCount / 2000)) == (_episodeCount / 2000)){ //diminishing stochastic exploration
	if(_totalEpisodes < 15000){
		if(rand() % 5 == 4){
			/*if(rand() % 2 == 0){
				CurrentAction = _getStochasticOptimalAction();
			}
			else*/
			CurrentAction = (Action)(rand() % NUM_ACTIONS);
		}
	}
	else{
		CurrentAction = _searchForOptimalAction(world, missiles, 3);
	}
	

	//map the action into outputs
	_takeAction(CurrentAction);

	//some metalogic stuff: random restarts and force agent to move if in same place too long
	//TODO: this member represents bad coupling
	agent.sufferedCollision = false;
	_episodeCount++;
	_totalEpisodes++;

	//testing: print the neural net weights
	//if(_episodeCount % 100 == 1){
		//_qNet.PrintWeights();
	//}
}

/*
This implements the vanilla/classical q-value update scheme, for which the agent only receives a reward (-1, +10, etc)
at terminal states (collision, falls off cliff, reaches goal, etc). Its tricky getting this discretized scheme working
in the real-valued q-world, but it randomly started working when I imposed the constraint to only update neural network
values when the reward function returns some non-zero number, aka when it reaches a terminal state. Hence the bias leading
to divergence, since the agent spends a lot of time running into things (negative reward) before it ever finds the goal.

This works, albeit intermittently. The intermittance so far seems to be a product of whether or not the agent
ever finds the goal. If it doesn't, it continually experiences negative rewards, often pushing the network weights
to negative infinity (this could likely be prevented with network countermeasures like weight decay).
Likewise, if it goes without finding the goal for a while, running into things, it seems to unlearn its goal-finding behavior.
A version of Pocket could be used, such as storing the best neural net weights (best policy) for the maximum reward
for some measure of cumulative reward.

Either way, lots of heuristics could help this perform better. But toying with local hacks isn't all that interesting.

Parameters under which thie method has usually worked well (these found a good policy only about 1:6 times)
	neural net hidden dimension = STATE_DIMENSION
	eta=0.1
	momentum=0.5
	_gamma = 0.9
	#define EXTERNAL_REWARD_GOAL 2
	#define EXTERNAL_REWARD_VISITED -0.1
	#define EXTERNAL_REWARD_COLLISION -3
	search_horizon = 3
10/24: Under these params, the agent shows excellent convergence, especially with the action-search-policy. It still exhibits unlearning,
which is evident when the agent goes on policy during a particularly bad episode of collisions or traps. This could be remedied with a lower
learning rate (0.05 works great), by taking measures of the agent's highest performance and 'pocketing' the network weights at that stage, and so on.

10/27: A three layer net with 2 hidden nodes per layer (a kind of dim-reduction) also works well. No formal result on if its better.

*/
void Q2Agent::ClassicalUpdate(const World* world, const vector<Missile>& missiles)
{
	int action;
	double maxQ, qTarget, prevEstimate;
	Action optimalAction = ACTION_UP;

	//Update agent's current state and state history for all possible actions
	_updateCurrentActionStates(world, missiles);

	/*params have worked for all of these ranges: eta=[0.01-0.08], gamma=[0.8-0.99]
	It would be nice to figure out the param relationships to find the optimal settings.
	*/
	_qNet.SetEta(0.08);
	//Theoretically momentum may be bad for online learning: since the inputs are so correlated, momentum causes unlearning (too much local adaptation).
	_qNet.SetMomentum(0.2);
	_qNet.SetWeightDecay(0.001); //Arbitrary. Larger (~0.01) seems to indeed improve results, but be wary of it encouraging traps.
	_gamma = 0.9;

	//classify the new current-state across all action-nets 
	for(action = 0, maxQ = -10000000; action < NUM_ACTIONS; action++){
		//classify the state we just entered, given the previous action
		_qNet.Classify(_getCurrentState((Action)action));
		//cout << GetActionStr(action) << "\t" << _qNet.GetOutputs()[0].Output << endl;
		_currentActionValues[action] = _qNet.GetOutputs()[0].Output;
		//track the max action available in current state
		if(_qNet.GetOutputs()[0].Output > maxQ){
			maxQ = _qNet.GetOutputs()[0].Output;
			optimalAction = (Action)action;
		}
	}
	
	//get the target q factor from the experienced reward given the last action
	//qTarget = _getCurrentRewardValue_Manual(world, missiles) + _gamma * maxQ;
	//double reward = _getCurrentRewardValue_Logistic(world, missiles);
	double reward = _getCurrentRewardValue_Terminal(world, missiles);
	//cout << "reward: " << reward << endl;
	qTarget = reward + _gamma * maxQ;
	//cout << "QTARGET: " << qTarget << endl;
	_epochReward += qTarget;
	//cout << "qTarget: " << qTarget << " maxQ: " << maxQ << endl;

	//a possible heuristic: allow the agent to continue learning, but slowly, after converging
	//if(_totalEpisodes > 10000){
	//	_qNet.SetEta(0.001);
	//}

	//if(reward != 0.0 && _totalEpisodes < 20000){
	if(_totalEpisodes < 10000){
		//for(int i = 0; i < 3; i++){
		//cout << "currentaction " << (int)CurrentAction << " qnets.size()=" << _qNets.size() << endl;
		//backpropagate the error and update the network weights for the last action (only)
		const vector<double>& previousState = _getPreviousState((Action)CurrentAction);
		_qNet.Classify(previousState); //the net must be re-clamped to the previous state's signals
		prevEstimate = _qNet.GetOutputs()[0].Output;
		cout << "reward: " << reward << "    prev estimate: " << prevEstimate << endl;
		_qNet.BackpropagateError(previousState, qTarget);
		_qNet.UpdateWeights(previousState, qTarget);
		//cout << "44" << endl;
		if(_totalEpisodes > 10000){
			//record this example; this is useful for both replay-based learning and for data analysis
			_recordExample(_getPreviousState((Action)CurrentAction), qTarget, prevEstimate, CurrentAction);
		}
		//}
	}

	//take the action with the highest q-value
	//LastAction = CurrentAction;
	CurrentAction = optimalAction;

	//randomize the action n% of the time
	//if(rand() % (1 + (_episodeCount / 2000)) == (_episodeCount / 2000)){ //diminishing stochastic exploration
	if(_totalEpisodes < 10000){
		if(rand() % 5 == 4){
			/*
			if(rand() % 2 == 0)
				CurrentAction = _getStochasticOptimalAction();
			else
				CurrentAction = (Action)(rand() % NUM_ACTIONS);
			*/
			CurrentAction = (Action)(rand() % NUM_ACTIONS);
		}
		/*Experimental action policy: let the agent take the action with the best long-term value;
		this does NOT effect learning, which is still done wrt actions only t=1 step apart.
		Observations: This works well, but it has to be tuned kind of like a learning rate. For instance,
		whilst the agent is initially learning, its estimates are poor; hence estimating several steps ahead
		still contains a great deal of uncertainty. This can lead to chaotic choices, where the agent mistakenly
		assumes some direction is good several steps ahead, and divergently veers off in poor directions.
		Hence, this works, but could use some sort of damping parameter, such as gradually increasing the
		'horizon' parameter to the search function.
		
		There's also much to be said of the 'heavy shoes' perspective, requiring the agent to learn under difficult
		circumstances (difficult == highly local estimates), and then removing those 'heavy shoes' once the agent
		has adequately learned and allowing it to search for optimal semi-local actions using its state estimator.
		
		else{
			CurrentAction = _searchForOptimalAction(world, missiles, 2);		
		}
		*/
	}
	else{
		//An alternative, more-complex action-policy compared to basic e-greedy policies
		//Results: This works nicely (0% collision rate), usually only in compliment with shutting off
		//backpropagation at the same epoch/episode when e-greedy is shut off.
		CurrentAction = _searchForOptimalAction(world, missiles, 3);
	}

	//map the action into outputs
	_takeAction(CurrentAction);

	//some metalogic stuff: random restarts and force agent to move if in same place too long
	//TODO: this member represents bad coupling
	agent.sufferedCollision = false;
	_episodeCount++;
	_totalEpisodes++;

	//testing: print the neural net weights
	//if(_episodeCount % 100 == 1){
		//_qNet.PrintWeights();
	//}
}

/*
Same as Classical Q-learning, but with with no max operator over actions of the successor state.

Works great!
*/
void Q2Agent::SarsaUpdate(const World* world, const vector<Missile>& missiles)
{
	int action;
	double maxQ, qTarget, prevEstimate;
	Action optimalAction = ACTION_UP;

	//Update agent's current state and state history for all possible actions
	_updateCurrentActionStates(world, missiles);

	/*params have worked for all of these ranges: eta=[0.01-0.08], gamma=[0.8-0.99]
	It would be nice to figure out the param relationships to find the optimal settings.
	*/
	_qNet.SetEta(0.02);
	//Theoretically momentum may be bad for online learning: since the inputs are so correlated, momentum causes unlearning (too much local adaptation).
	_qNet.SetMomentum(0.5);
	_qNet.SetWeightDecay(0.001); //arbitrary, not something I played with much
	_gamma = 0.9;

	//classify the new current-state across all action-nets 
	optimalAction = (Action)(rand() % NUM_ACTIONS);
	for(action = 0, maxQ = -10000000; action < NUM_ACTIONS; action++){
		//classify the state we just entered, given the previous action
		_qNet.Classify(_getCurrentState((Action)action));
		//cout << GetActionStr(action) << "\t" << _qNet.GetOutputs()[0].Output << endl;
		_currentActionValues[action] = _qNet.GetOutputs()[0].Output;
		//track the max action available in current state
		if(_qNet.GetOutputs()[0].Output > maxQ){
			maxQ = _qNet.GetOutputs()[0].Output;
			optimalAction = (Action)action;
		}
	}
	
	//get the target q factor from the experienced reward given the last action
	double reward = _getCurrentRewardValue_Terminal(world, missiles);
	//cout << "reward: " << reward << endl;
	qTarget = reward + _gamma * _currentActionValues[CurrentAction];
	//cout << "QTARGET: " << qTarget << endl;
	_epochReward += qTarget;
	//cout << "qTarget: " << qTarget << " maxQ: " << maxQ << endl;

	//if(reward != 0.0 && _totalEpisodes < 20000){
	if(_totalEpisodes < 10000){
		//for(int i = 0; i < 3; i++){
		//cout << "currentaction " << (int)CurrentAction << " qnets.size()=" << _qNets.size() << endl;
		//backpropagate the error and update the network weights for the last action (only)
		const vector<double>& previousState = _getPreviousState((Action)CurrentAction);
		_qNet.Classify(previousState); //the net must be re-clamped to the previous state's signals
		prevEstimate = _qNet.GetOutputs()[0].Output;
		cout << "reward: " << reward << "    prev estimate: " << prevEstimate << endl;
		_qNet.BackpropagateError(previousState, qTarget);
		_qNet.UpdateWeights(previousState, qTarget);
		//cout << "44" << endl;
		if(_episodeCount > 100){
			//record this example; this is useful for both replay-based learning and for data analysis
			_recordExample(_getPreviousState((Action)CurrentAction), qTarget, prevEstimate, CurrentAction);
		}
		//}
	}

	//take the action with the highest q-value
	//LastAction = CurrentAction;
	CurrentAction = optimalAction;
/*
	//SARSA normally learns on-policy, so no e-greedy action selection; not sure if this is sound in real-domain though
	if(_totalEpisodes < 10000){
		if(rand() % 5 == 4){	
			CurrentAction = (Action)(rand() % NUM_ACTIONS);
		}
	}
	*/
	
	
	if(_totalEpisodes > 10000){
		//An alternative, more-complex action-policy compared to basic e-greedy policies
		//Results: This works nicely (0% collision rate), usually only in compliment with shutting off
		//backpropagation at the same epoch/episode when e-greedy is shut off.
		CurrentAction = _searchForOptimalAction(world, missiles, 3);
	}
	

	//map the action into outputs
	_takeAction(CurrentAction);

	//some metalogic stuff: random restarts and force agent to move if in same place too long
	//TODO: this member represents bad coupling
	agent.sufferedCollision = false;
	_episodeCount++;
	_totalEpisodes++;

	//testing: print the neural net weights
	//if(_episodeCount % 100 == 1){
		//_qNet.PrintWeights();
	//}
}


/*
An alternative to basic e-greedy action policies. Excluding other details or information that
inform the choice of policy, this policy structure allows the agent to prognosticate a little bit
about the values of actions within its local space. For instance, the immediate value may not simply
be the estimated state-value resulting from going right, but rather the agent may check for several
state values ahead of taking the 'right' action, and then takes the value of 'right' as being the max
of those values for some k-steps of prognostication. The agent could also use a weighted-sum of these
actions, or might even perform a recursive search (for instance, within a radius of size k) for the direction
of the state with the highest value.

In sum, once the agent is trained, the implementor can use the learned value structure however they wish, to maximize
the agent's behavior.

Once the agent is trained, its perfectly fair to implement any state-value-estimation strategy desired via
its state-estimator and state-value-estimator (the q-net).

@searchHorizon: The local horizon of the agent's sensors, in terms of number of steps ahead it may estimate states
*/
Action Q2Agent::_searchForOptimalAction(const World* world, const vector<Missile>& missiles, const int searchHorizon)
{
	//Simple cross search: let the agent simulate taking a given action repeatedly for k steps; its estimate
	//of the immediate value of the action is then the max of those values. This pushes the agent toward better values.
	//Note that this also gives globally optimal behavior optimal, since the "cross" of the search is not stationary, but is
	//updated after an action is taken, hence leading the agent to search most local values.
	bool collision;
	int i, j, action;
	double xHeading, yHeading, destX, destY, valueEstimate;
	double maxQ = -1000000;
	Action maxAction = ACTION_DOWN;
	vector<double> tempState;
	
	tempState.resize(STATE_DIMENSION,0);
	
	//Simulate each action repeatedly for k steps
	for(action = 0; action < NUM_ACTIONS; action++){
		//get the velocity/heading state-specific values
		switch(action){
			case ACTION_UP:
				xHeading = 0;
				yHeading = 1.0;
				break;
			case ACTION_RIGHT:
				xHeading = 1.0;
				yHeading = 0;
				break;
			case ACTION_DOWN:
				xHeading = 0;
				yHeading = -1.0;
				break;
			case ACTION_LEFT:
				xHeading = -1.0;
				yHeading = 0;
				break;
			default:
				cout << "ERROR action not found in _updateState()" << endl;
				break;				
		}

		//estimate the action-states resulting after taking this action i times or until an obstacle is encountered
		for(i = 1, collision = false; i <= searchHorizon && !collision; i++){
			//estimate of where the agent will be, given this action
			destX = (double)agent.x + xHeading * (double)i;
			destY = (double)agent.y + yHeading * (double)i;
			//only estimate up until an obstacle/off-map position is encountered
			collision = !world->IsValidPosition((int)destX, (int)destY) || world->GetCell((int)destX, (int)destY).isObstacle; //TODO: dangerous short-circuit logic
			if(!collision){
				/*
				//estimate max over all *available* actions (headings) in each possible subsequent state
				for(j = 0; j < NUM_ACTIONS; j++){
					switch(j){
						case ACTION_UP:
							_estimateSubsequentState(0, 1, destX, destY, world, missiles, tempState);
						break;
						case ACTION_RIGHT:
							_estimateSubsequentState(1, 0, destX, destY, world, missiles, tempState);
						break;
						case ACTION_DOWN:
							_estimateSubsequentState(0, -1, destX, destY, world, missiles, tempState);
						break;
						case ACTION_LEFT:
							_estimateSubsequentState(-1, 0, destX, destY, world, missiles, tempState);
						break;
						default:
							cout << "CASE NOT FOUND IN _searchForOptimalAction()" << endl;
						break;
					}

					_qNet.Classify(tempState);
					//the maximization step, over the states resulting from executing this action i times
					valueEstimate = _qNet.GetOutputs()[0].Output;
					if(valueEstimate > maxQ){
						maxQ = valueEstimate;
						maxAction = (Action)action;
					}				
				}
				*/
				
				//estimate the ith next-state
				_estimateSubsequentState(xHeading, yHeading, destX, destY, world, missiles, tempState);						
				//use the state estimate to drive the trained q-net
				_qNet.Classify(tempState);
				//the maximization step, over the states resulting from executing this action i times
				valueEstimate = _qNet.GetOutputs()[0].Output;
				if(valueEstimate > maxQ){
					maxQ = valueEstimate;
					maxAction = (Action)action;
				}
				
				
				cout << GetActionStr((int)action) << " estimate: " << valueEstimate << endl;
			}
		}
	}
	
	cout << "Executing max action (of non-collision actions) " << GetActionStr((int)maxAction) << " with value: " << maxQ << endl;

	return maxAction;
}


/*
THIS IS JUST EXPERIMENTAL. I have no idea if this is a correct Baird-Advantage-Update interpretation.

This works, but it isn't clear why. I hesitate to trust this hasty implementation, since the error
is backpropagated wrt to the max action-state pair in the previous state, which may or may not represent
the action-state pair that brought us to the successor state. In most cases, they will be the same, since
we usually choose the max-value action in each state; but for epsilon-percent of the time, we don't, hence
the update is wrt action-state pair that wasn't executed. It seems like this only works because the majority
of the time the update is not inconsistent wrt the previous state-action pair.
*/
void Q2Agent::AdvantageUpdate(const World* world, const vector<Missile>& missiles)
{
	int action;
	double maxQ, qTarget, prevEstimate, prevMaxQ;
	Action optimalAction = ACTION_UP;
	Action prevOptimalAction = ACTION_UP;

	//Update agent's current state and state history for all possible actions
	_updateCurrentActionStates(world, missiles);

	/*params have worked for all of these ranges: eta=[0.01-0.08], gamma=[0.8-0.99]
	It would be nice to figure out the param relationships to find the optimal settings.
	*/
	_qNet.SetEta(0.02);
	_qNet.SetMomentum(0.1);
	_gamma = 0.9;

	//classify the new current-state across all action-nets 
	for(action = 0, maxQ = -10000000; action < NUM_ACTIONS; action++){
		//classify the state we just entered, given the previous action
		_qNet.Classify(_getCurrentState((Action)action));
		//cout << GetActionStr(action) << "\t" << _qNet.GetOutputs()[0].Output << endl;
		_currentActionValues[action] = _qNet.GetOutputs()[0].Output;
		//track the max action available in current state
		if(_qNet.GetOutputs()[0].Output > maxQ){
			maxQ = _qNet.GetOutputs()[0].Output;
			optimalAction = (Action)action;
		}
	}

	double reward = _getCurrentRewardValue_Terminal(world, missiles);
	//cout << "reward: " << reward << endl;
	qTarget = reward + _gamma * maxQ;
	//cout << "QTARGET: " << qTarget << endl;
	_epochReward += qTarget;

	//classify the new current-state across all action-nets 
	for(action = 0, prevMaxQ = -10000000; action < NUM_ACTIONS; action++){
		//classify the state we just entered, given the previous action
		_qNet.Classify(_getPreviousState((Action)action));
		//cout << GetActionStr(action) << "\t" << _qNet.GetOutputs()[0].Output << endl;
		_currentActionValues[action] = _qNet.GetOutputs()[0].Output;
		//track the max action available in current state
		if(_qNet.GetOutputs()[0].Output > prevMaxQ){
			prevMaxQ = _qNet.GetOutputs()[0].Output;
			prevOptimalAction = (Action)action;
		}
	}

	if(_totalEpisodes < 20000){
		//cout << "currentaction " << (int)CurrentAction << " qnets.size()=" << _qNets.size() << endl;
		//backpropagate the error and update the network weights for the last action (only)
		const vector<double>& previousState = _getPreviousState((Action)prevOptimalAction);
		_qNet.Classify(previousState); //the net must be re-clamped to the previous state's signals
		prevEstimate = _qNet.GetOutputs()[0].Output;
		cout << "reward: " << reward << "    prev estimate: " << prevEstimate << endl;
		_qNet.BackpropagateError(previousState, qTarget);
		_qNet.UpdateWeights(previousState, qTarget);

		//cout << "44" << endl;
		if(_episodeCount > 100){
			//record this example; this is useful for both replay-based learning and for data analysis
			_recordExample(_getPreviousState((Action)CurrentAction), qTarget, prevEstimate, CurrentAction);
		}
	}

	//take the action with the highest q-value
	//LastAction = CurrentAction;
	CurrentAction = optimalAction;

	//randomize the action n% of the time
	//if(rand() % (1 + (_episodeCount / 2000)) == (_episodeCount / 2000)){ //diminishing stochastic exploration
	if(_totalEpisodes < 20000){
		if(rand() % 5 == 4){
			CurrentAction = (Action)(rand() % NUM_ACTIONS);
		}
	}
	else{
		CurrentAction = _searchForOptimalAction(world, missiles, 2);
	}

	//map the action into outputs
	_takeAction(CurrentAction);

	//some metalogic stuff: random restarts and force agent to move if in same place too long
	//TODO: this member represents bad coupling
	agent.sufferedCollision = false;
	_episodeCount++;
	_totalEpisodes++;
}


/*
Virtually the same as classical q-learning, however instead of maxQ{a in Action}(Q(s',a)),
this averages over all of the q-values in the successor state. Theoretically this should give a milder,
more conservative, possibly more convergent agent that favors states with better average value over extreme
actions with only one good value. It just seemed like a simply modification to basic q-learning to try.
This method could possibly operate better in non-stationary environments, since the agent might be more
averse/defensive compared with ClassicalQ-learning. But this is a big 'perhaps', and requires more
rigorous mathematical justification. The aggressiveness of Classical Q-Learning can be coupled with 
a conservative action-policy, which is likely a more robust way to engineer the agent's behavior; and likewise
doing so preserves the optimal/precise information given by aggressive learning styles.

This works just fine. Possibly better than Classical-Update. This means there are probably many different ways to
1) Play with the basic q-value recursion for better performance and interesting behavior/policy outcomes
2) Define the recursion to select actions optimizing different expected values per action (eg, using risk analysis)

The search-based action policy has different behavior here, and may want for different horizon values. The agent always
reaches the goal, but occasionally takes drastically conservative actions very close to the goal, such as nearly
reaching the goal then bailing to the opposite side of the world, before returning directly (optimally/very quickly) to
the goal with precision. This may be the action policy, or noise in the learned weights. More training episodes helped
overcome some of this, but it would still occur. It just seems like the agent occasionally favors very low-density regions
for a while, behaving very conservatively, before finding the goal very quickly.
Constraining the agent to only learn non-zero reward values eliminated the problem, so this looks like another over-learning
phenomena.
*/
void Q2Agent::AverageUpdate(const World* world, const vector<Missile>& missiles)
{
	int action;
	double maxQ, sumQ, qTarget, prevEstimate;
	Action optimalAction = ACTION_UP;

	//Update agent's current state and state history for all possible actions
	_updateCurrentActionStates(world, missiles);

	/*
	Params have worked for all of these ranges: eta=[0.01-0.1], gamma=[0.8-0.99]
	It would be nice to figure out the param relationships to find the optimal settings.
	*/
	_qNet.SetEta(0.05);
	//Theoretically momentum may be bad for online learning: since the inputs are so correlated, momentum causes unlearning (too much local adaptation).
	_qNet.SetMomentum(0.5);
	_qNet.SetWeightDecay(0.001); //arbitrary, not something I played with much
	_gamma = 0.9;

	//classify the new current-state across all action-nets 
	for(action = 0, sumQ = 0, maxQ = -10000000; action < NUM_ACTIONS; action++){
		//classify the state we just entered, given the previous action
		_qNet.Classify(_getCurrentState((Action)action));
		//cout << GetActionStr(action) << "\t" << _qNet.GetOutputs()[0].Output << endl;
		_currentActionValues[action] = _qNet.GetOutputs()[0].Output;
		sumQ += _qNet.GetOutputs()[0].Output;
		//track the max action available in current state
		if(_qNet.GetOutputs()[0].Output > maxQ){
			maxQ = _qNet.GetOutputs()[0].Output;
			optimalAction = (Action)action;
		}
	}
	
	//get the target q factor from the experienced reward given the last action
	//qTarget = _getCurrentRewardValue_Manual(world, missiles) + _gamma * maxQ;
	//double reward = _getCurrentRewardValue_Logistic(world, missiles);
	double reward = _getCurrentRewardValue_Terminal(world, missiles);
	//cout << "reward: " << reward << endl;
	//qTarget = reward + _gamma * maxQ;
	qTarget = reward + _gamma * (sumQ / (double)NUM_ACTIONS);
	//cout << "QTARGET: " << qTarget << endl;
	_epochReward += qTarget;
	//cout << "qTarget: " << qTarget << " maxQ: " << maxQ << endl;

	//constraining updates seems to help this method behave better on-policy
	if(reward != 0.0 && _totalEpisodes < 20000){
		//cout << "currentaction " << (int)CurrentAction << " qnets.size()=" << _qNets.size() << endl;
		//backpropagate the error and update the network weights for the last action (only)
		const vector<double>& previousState = _getPreviousState((Action)CurrentAction);
		_qNet.Classify(previousState); //the net must be re-clamped to the previous state's signals
		prevEstimate = _qNet.GetOutputs()[0].Output;
		cout << "reward: " << reward << "    prev estimate: " << prevEstimate << endl;
		_qNet.BackpropagateError(previousState, qTarget);
		_qNet.UpdateWeights(previousState, qTarget);
		//cout << "44" << endl;
		if(_episodeCount > 100){
			//record this example; this is useful for both replay-based learning and for data analysis
			_recordExample(_getPreviousState((Action)CurrentAction), qTarget, prevEstimate, CurrentAction);
		}
	}

	//take the action with the highest q-value
	//LastAction = CurrentAction;
	CurrentAction = optimalAction;

	//randomize the action n% of the time
	//if(rand() % (1 + (_episodeCount / 2000)) == (_episodeCount / 2000)){ //diminishing stochastic exploration
	if(_totalEpisodes < 20000){
		if(rand() % 5 == 4){
			/*
			if(rand() % 2 == 0)
				CurrentAction = _getStochasticOptimalAction();
			else
				CurrentAction = (Action)(rand() % NUM_ACTIONS);
			*/
			CurrentAction = (Action)(rand() % NUM_ACTIONS);
		}
	}
	else{
		CurrentAction = _searchForOptimalAction(world, missiles, 2);
	}

	//map the action into outputs
	_takeAction(CurrentAction);

	//some metalogic stuff: random restarts and force agent to move if in same place too long
	//TODO: this member represents bad coupling
	agent.sufferedCollision = false;
	_episodeCount++;
	_totalEpisodes++;

	//testing: print the neural net weights
	//if(_episodeCount % 100 == 1){
		//_qNet.PrintWeights();
	//}
}



//gets the ordinal distance between two doubles
double Q2Agent::_absDiff(double d1, double d2)
{
	double d = d1 - d2;

	if(d < 0){
		return -d;
	}
	return d;
}

/*
Learn q-values only when terminal states are reached: collisions, goals, death, etc.


void Q2Agent::EpochalUpdate(const World* world, const vector<Missile>& missiles)
{
	int i;
	double maxQ, qTarget;
	Action optimalAction = ACTION_UP;

	//Update agent's current state and state history
	_updateCurrentActionStates(world, missiles);

	//classify the new current-state across all action-nets 
	for(i = 0, maxQ = -10000000; i < _qNets.size(); i++){
		//classify the state we just entered, given the previous action
		_qNets[i].Classify(_getCurrentState((Action)i));
		//cout << "classified: " << _qNets[i].GetOutputs()[0].Output << endl;
		//track the max action available in current state
		if(_qNets[i].GetOutputs()[0].Output > maxQ){
			maxQ = _qNets[i].GetOutputs()[0].Output;
			optimalAction = (Action)i;
		}
	}

	//get the target q factor from the experienced reward given the last action
	qTarget = _getCurrentRewardValue_Manual(world, missiles) + _gamma * maxQ;
	//cout << "QTARGET: " << qTarget << endl;
	_epochReward += qTarget;

	//update nets only when some terminal event happens
	if(agent.sufferedCollision || world->GetCell(agent.x, agent.y).isGoal){
		//cout << "currentaction " << (int)CurrentAction << " qnets.size()=" << _qNets.size() << endl;
		//backpropagate the error and update the network weights for the last action (only)
		_qNets[(int)CurrentAction].Classify(_getPreviousState((Action)CurrentAction)); //the net must be re-clamped to the previous state inputs and signals
		//cout << "prev estimate: " << _qNets[(int)CurrentAction].GetOutputs()[0].Output << endl;
		_qNets[(int)CurrentAction].BackpropagateError(_getPreviousState((Action)CurrentAction), qTarget);
		_qNets[(int)CurrentAction].UpdateWeights(_getPreviousState((Action)CurrentAction), qTarget);
		//cout << "44" << endl;
	}

	//record this example
	//_recordExample(_getPreviousState((Action)CurrentAction), qTarget, _qNets[(int)CurrentAction].GetOutputs()[0].Output, CurrentAction);
	
	//take the action with the highest q-value
	//LastAction = CurrentAction;
	CurrentAction = optimalAction;

	//randomize the action n% of the time
	//if(rand() % (1 + (_episodeCount / 2000)) == (_episodeCount / 2000)){ //diminishing stochastic exploration
	if((rand() % 5) == 4 && _episodeCount < 100000){
		if(rand() % 2 == 0)
			CurrentAction = _getStochasticOptimalAction();
		else
			CurrentAction = (Action)(rand() % NUM_ACTIONS);
	}

	//map the action into outputs
	_takeAction(CurrentAction);

	//some metalogic stuff: random restarts and force agent to move if in same place too long
	//TODO: this member represents bad coupling
	agent.sufferedCollision = false;
	_episodeCount++;
	_totalEpisodes++;

	//testing: print the neural net weights
	if(_episodeCount % 100 == 1){
		for(i = 0; i < _qNets.size(); i++){
			cout << "Action net " << GetActionStr((Action)i) << endl;
			_qNets[i].PrintWeights();
		}
	}
}
*/

/*
For this experiment, only update the agent's q-networks when the agent reaches a terminal 
state, or when the error term is very large.

Result: Failure. There may still be potential in something like this, a heuristic for performing
updates only when significant events occur. But this implementation was too divergent. 

void Q2Agent::DiscriminativeUpdate(const World* world, const vector<Missile>& missiles)
{
	int i;
	double maxQ, qTarget;
	Action optimalAction = ACTION_UP;

	//Update agent's current state and state history
	_updateCurrentActionStates(world, missiles);

	//classify the new current-state across all action-nets 
	for(i = 0, maxQ = -10000000; i < _qNets.size(); i++){
		//classify the state we just entered, given the previous action
		_qNets[i].Classify(_getCurrentState((Action)i));
		//cout << "classified: " << _qNets[i].GetOutputs()[0].Output << endl;
		//track the max action available in current state
		if(_qNets[i].GetOutputs()[0].Output > maxQ){
			maxQ = _qNets[i].GetOutputs()[0].Output;
			optimalAction = (Action)i;
		}
	}

	//get the target q factor from the experienced reward given the last action
	qTarget = _getCurrentRewardValue_Manual(world, missiles) + _gamma * maxQ;
	//cout << "QTARGET: " << qTarget << endl;
	_epochReward += qTarget;
	//cout << "qTarget: " << qTarget << " maxQ: " << maxQ << endl;

	//cout << "currentaction " << (int)CurrentAction << " qnets.size()=" << _qNets.size() << endl;
	//backpropagate the error and update the network weights for the last action (only)
	_qNets[(int)CurrentAction].Classify(_getPreviousState((Action)CurrentAction)); //the net must be re-clamped to the previous state inputs and signals
	if(_absDiff(_qNets[(int)CurrentAction].GetOutputs()[0].Output, qTarget) > 10.0 || (rand() % 10 == 9)){
		//cout << "prev estimate: " << _qNets[(int)CurrentAction].GetOutputs()[0].Output << endl;
		_qNets[(int)CurrentAction].BackpropagateError(_getPreviousState((Action)CurrentAction), qTarget);
		_qNets[(int)CurrentAction].UpdateWeights(_getPreviousState((Action)CurrentAction), qTarget);
		//cout << "44" << endl;
	}

	//take the action with the highest q-value
	CurrentAction = optimalAction;

	//randomize the action n% of the time
	//if(rand() % (1 + (_episodeCount / 2000)) == (_episodeCount / 2000)){ //diminishing stochastic exploration
	if((rand() % 5) == 4 && _episodeCount < 10000){
		if(rand() % 2 == 0)
			CurrentAction = _getStochasticOptimalAction();
		else
			CurrentAction = (Action)(rand() % NUM_ACTIONS);
	}

	//map the action into outputs
	_takeAction(CurrentAction);

	//some metalogic stuff: random restarts and force agent to move if in same place too long
	//TODO: this member represents bad coupling
	agent.sufferedCollision = false;
	_episodeCount++;
	_totalEpisodes++;

	//testing: print the neural net weights
	if(_episodeCount % 100 == 1){
		for(i = 0; i < _qNets.size(); i++){
			cout << "Action net " << GetActionStr((Action)i) << endl;
			_qNets[i].PrintWeights();
		}
	}
}
*/

/*
Loose testing of offline learning just to see if if can work: run the agent for 
a few thousand or tens of thousands of episodes, recording all of its learned rewards.
Then using those stored examples, train the networks offline to see if that can help overcome
the locality issues with using neural nets for q-learning.

Results: Not good. Agent more or less just exercised whatever last bits of local information it had learned.
There was no global learning; mathematically, the error when training the neural nets on all the training data 
never decreased to some very small positive number, instead reduced to around 3-5 (the q value error), and never
decreased further. This showed that the dependence of the network on the weights of previous iterations (through the
q-estimates in the target value) results in non-sensical, highly local learning.
	1) local learning may be useful, if coupled with some other architecture
	2) could change the reward/learning scheme; for instance, only update when large error terms occur, or only when
		the agent reaches some terminal event (collision, death, or goal-reached).

This experiment was very useful because it showed the instability/undecidability which the agent is trying to learn. Since the
data itself does not globally converge under any architecture, the function itself must not be learnable/differentiable. Its
very likely to be a piecewise function, etc.


void Q2Agent::OfflineUpdate(const World* world, const vector<Missile>& missiles)
{
	int pid;
	int trainingEpisodes = 10000;
	vector<vector<double> > trainingData;

	//run the regular Update() method for k-thousand times
	if(_totalEpisodes < trainingEpisodes){
		Update(world, missiles);
		return;
	}
	
	if(_totalEpisodes == trainingEpisodes){
		agent.xVelocity = 0; //halt the agent
		agent.yVelocity = 0;

		//fire a python proc to split the log into data for each network
		if(pid = fork()){ //parent waits for python proc to complete
			//wait for child python proc to return, meaning the per-action training data has been written
			int status;
			waitpid(pid,&status,0);

			//don't clear the nets: likely, the previous values of the network will be locally minimized enough to avoid divergence?
			//TODO: try weight randomization 
			for(int i = 0; i < _qNets.size(); i++){
				string path;
				switch(i){
					case ACTION_UP:
						path = "up.txt";
					break;
					case ACTION_DOWN:
						path = "down.txt";
					break;
					case ACTION_LEFT:
						path = "left.txt";
					break;
					case ACTION_RIGHT:
						path = "right.txt";
					break;
				}

				_qNets[i].AssignRandomWeights();
				_qNets[i].ReadCsvDataset(path, trainingData);
				_qNets[i].BatchTrain(trainingData, 0.04, 0.0); // use small eta for stability
				//_qNets[i].SaveNetwork(path);
			}
		}
		else{
			execlp("python", "python", "History2Datasets.py", "history.txt", (char*) NULL);
		}
		_totalEpisodes++;
	}
	//else, global training complete: run the agent using the globally-learnt weights, with no further learning, and see what happens
	else{
		int i;
		double maxQ, qTarget;
		Action optimalAction = ACTION_UP;

		//Update agent's current state and state history
		_updateCurrentActionStates(world, missiles);

		//classify the new current-state across all action-nets 
		for(i = 0, maxQ = -10000000; i < _qNets.size(); i++){
			//classify the state we just entered, given the previous action
			_qNets[i].Classify(_getCurrentState((Action)i));
			//track the max action available in current state
			if(_qNets[i].GetOutputs()[0].Output > maxQ){
				maxQ = _qNets[i].GetOutputs()[0].Output;
				optimalAction = (Action)i;
			}
		}

		//get the target q factor from the experienced reward given the last action
		qTarget = _getCurrentRewardValue_Manual(world, missiles) + _gamma * maxQ;
		//cout << "QTARGET: " << qTarget << endl;
		_epochReward += qTarget;

		//take the action with the highest q-value
		CurrentAction = optimalAction;

		//randomize the action n% of the time
		if((rand() % 5) == 4 && _episodeCount < 10000){
			if(rand() % 2 == 0)
				CurrentAction = _getStochasticOptimalAction();
			else
				CurrentAction = (Action)(rand() % NUM_ACTIONS);
		}

		//map the action into outputs
		_takeAction(CurrentAction);

		//some metalogic stuff: random restarts and force agent to move if in same place too long
		//TODO: this member represents bad coupling
		agent.sufferedCollision = false;
		_episodeCount++;
		_totalEpisodes++;
	}
}
*/

/*
The same implementation as vanilla Update() above, but the agent pushes each (state,qEstimate,qTarget) to
a small, short-term batch of experiences. It then stochastically removes a random experience to train on.

void Q2Agent::MinibatchUpdate(const World* world, const vector<Missile>& missiles)
{
	int i;
	double maxQ, qTarget;
	Action optimalAction = ACTION_UP;

	if(_episodeCount <= 2000){
		Update(world,missiles);

		//drop eta once in minibatch mode
		for(i = 0; i < _qNets.size(); i++){
			_qNets[i].SetEta(0.05);
		}

		return;
	}

	//Update agent's current state and state history
	_updateCurrentActionStates(world, missiles);

	//enlarge the batch until size equals BATCH_SIZE (thus, no overwriting of experiences until size > BATCH_SIZE
	if(_batch.size() < BATCH_SIZE){
		_batch.resize(_batch.size() + 1);
		_batchIndex = _batch.size() - 1;
	}
	//save the current state in the batch
	for(i = 0; i < _stateHistory[_t][(int)CurrentAction].size(); i++){
		_batch[_batchIndex].BatchedState[i] = _stateHistory[_t][(int)CurrentAction][i];
	}
	_batch[_batchIndex].PerformedAction = CurrentAction;

	//cout << "11" << endl;
	//classify the new current-state across all action-nets 
	for(i = 0, maxQ = -10000000; i < _qNets.size(); i++){
		//classify the state we just entered, given the previous action
		_qNets[i].Classify(_getCurrentState((Action)i));
		//cout << "classified: " << _qNets[i].GetOutputs()[0].Output << endl;
		//track the max action available in current state
		if(_qNets[i].GetOutputs()[0].Output > maxQ){
			maxQ = _qNets[i].GetOutputs()[0].Output;
			optimalAction = (Action)i;
		}
	}
	//save current QEstimate
	_qNets[(int)CurrentAction].Classify(_getPreviousState((Action)CurrentAction));
	_batch[_batchIndex].QEstimate = _qNets[(int)CurrentAction].GetOutputs()[0].Output;
	//save the experienced reward as the qTarget in the batched experience
	_batch[_batchIndex].QTarget = _getCurrentRewardValue_Manual(world, missiles) + _gamma * maxQ;
	_epochReward += _batch[_batchIndex].QTarget;
	//save the action that brought us here
	_batch[_batchIndex].PerformedAction = CurrentAction;

	//update batch index to a new random location in which to store the next experience
	_batchIndex = rand() % _batch.size();
	Experience& exp = _batch[_batchIndex];
	_qNets[(int)exp.PerformedAction].Classify(exp.BatchedState); //the net must be re-clamped to the previous state inputs and signals
	_qNets[(int)exp.PerformedAction].BackpropagateError(exp.BatchedState, exp.QTarget);
	_qNets[(int)exp.PerformedAction].UpdateWeights(exp.BatchedState, exp.QTarget);
	
	
	//train over a collection from the batch
	//for(i = 0; i < 50; i++){
	//	Experience& exp = _batch[ rand() % _batch.size() ];
	//	_qNets[(int)exp.PerformedAction].Classify(exp.BatchedState); //the net must be re-clamped to the previous state inputs and signals
	//	_qNets[(int)exp.PerformedAction].BackpropagateError(exp.BatchedState, exp.QTarget);
	//	_qNets[(int)exp.PerformedAction].UpdateWeights(exp.BatchedState, exp.QTarget);
	//}


	//take the action with the highest q-value
	//LastAction = CurrentAction;
	CurrentAction = optimalAction;
	//randomize the action n% of the time
	//if(rand() % (1 + (_episodeCount / 2000)) == (_episodeCount / 2000)){ //diminishing stochastic exploration
	if((rand() % 5) == 4){
		if(rand() % 2 == 0)
			CurrentAction = _getStochasticOptimalAction();
		else
			CurrentAction = (Action)(rand() % NUM_ACTIONS);
	}

	//map the action into outputs
	_takeAction(CurrentAction);

	//some metalogic stuff: random restarts and force agent to move if in same place too long
	//TODO: this member represents bad coupling
	agent.sufferedCollision = false;
	_episodeCount++;

	
	//testing: print the neural net weights
	if(_episodeCount % 100 == 1){
		for(i = 0; i < _qNets.size(); i++){
			cout << "Action net " << GetActionStr((Action)i) << endl;
			_qNets[i].PrintWeights();
		}
	}
	

}
*/




void Q2Agent::_takeAction(Action nextAction)
{
	//map the action into some output
	switch(nextAction){
		case ACTION_UP:
				agent.xVelocity = 0.0;
				agent.yVelocity = 1.0;
			break;
		case ACTION_DOWN:
				agent.xVelocity = 0.0;
				agent.yVelocity = -1.0;
			break;
		case ACTION_LEFT:
				agent.xVelocity = -1.0;
				agent.yVelocity = 0.0;
			break;
		case ACTION_RIGHT:
				agent.xVelocity = 1.0;
				agent.yVelocity = 0.0;
			break;
		/*
		case ACTION_SHOOT:
			//nothing, world will respond to shot spawn; normally in game programming this would spawn an event
			cout << "SHOTS FIRED" << endl;
			break;
		
		case ACTION_IDLE:
				agent.xVelocity = 0.0;
				agent.yVelocity = 0.0;
			break;
		*/
		default:
				cout << "ERROR unknown Action: " << (int)CurrentAction << endl;
			break;
	}

}

const char* Q2Agent::GetActionStr(int i)
{
	switch((Action)i){
		case ACTION_UP:
			 return "UP";
			break;
		case ACTION_DOWN:
			 return "DOWN";
			break;
		case ACTION_LEFT:
			 return "LEFT";
			break;
		case ACTION_RIGHT:
			 return "RIGHT";
			break;
		/*
		case ACTION_UP:
			 return "UP";
			break;
		case ACTION_UP:
			 return "UP";	
			break;
		*/
		default:
			return "ERROR UNKNOWN ACTION";
			break;
	}
}

void Q2Agent::PrintState()
{
	const vector<double>& s = _getCurrentState(CurrentAction);
	cout << _totalEpisodes << " Epoch " << _epochCount << " Episode " <<  _episodeCount << " Agent (" << agent.x << "x," << agent.y << "y) " << " <xVel yVel cosGoal cosVisited obstDist goalDist>:" << endl;
	cout << agent.xVelocity << " " << agent.yVelocity << " " << s[SA_GOAL_COSINE] << " " << s[SA_RECENT_LOCATION_COSINE] << " "  << s[SA_COLLISION_PROXIMITY] << " " << endl;
	//cout << agent.xVelocity << " " << agent.yVelocity << " " << s[SA_GOAL_COSINE] << " " << s[SA_RECENT_LOCATION_COSINE] << " "  << s[SA_COLLISION_PROXIMITY] << " " << s[SA_GOAL_DIST] << endl;
	cout << "Recent Loc-avg: " << _locationAvg.first << "," << _locationAvg.second << "    Loc-ema: " << _locationEma.first << "," << _locationEma.second << endl;
 	cout << " Last Epoch reward: " << _lastEpochReward << endl;
	cout << "Action (just executed): [" << CurrentAction << "] " << GetActionStr(CurrentAction) << endl;
	cout << "Outputs:" << endl;
	for(int i = 0; i < _currentActionValues.size(); i++){
		if(i == CurrentAction){
			cout << GetActionStr(i) << "\t" << _currentActionValues[i] << " <-- current action " << endl;
		}
		else{
			cout << GetActionStr(i) << "\t" << _currentActionValues[i] << endl;
		}
	}
	//print the epoch measures; ideally all should decrease with training, and avg reward should increase
	cout << "Last epoch (" << _epochCount << " epoch / " << _totalEpisodes << " episodes) performance:  collision rate: " << ((double)((int)(_lastEpochCollisionRate * 1000)) / 100) << "%  #actions: " << _lastEpochActionCount;
	cout << endl;
}
