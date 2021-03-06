#include "QAgent.hpp"

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



Experience::Experience()
{
	BatchedState.resize(STATE_DIMENSION);
	QTarget = 0.0;
	QEstimate = 0.0;
	PerformedAction = ACTION_DOWN;
}

QAgent::QAgent(int initX, int initY)
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
	_t = 0; //time index

	//init the neural networks, one for each action
	_qNets.resize(NUM_ACTIONS);
	for(int i = 0; i < _qNets.size(); i++){
		//this could be custom/regularized in the future, where inputs and structure are customized to the action/state dependencies
		//adding lots of hidden nodes currently makes the agent more precise, but this may be a degenerate form of regularization (more neurons equals smoother approximations, more/smaller weights)
		//a good setup for local estimation: _qNets[i].BuildNet(2, STATE_DIMENSION, STATE_DIMENSION * 3, 1); //this is just generic, for testing;
		_qNets[i].BuildNet(2, STATE_DIMENSION, STATE_DIMENSION * 5, 1); //this is just generic, for testing;
		//set outputs to linear, since q-values are linear in some range after convergence like [-10.12-8.34]
		_qNets[i].SetHiddenLayerFunction(TANH);
		_qNets[i].SetOutputLayerFunction(LINEAR);
		_qNets[i].InitializeWeights();
		_qNets[i].SetEta(_eta);
		//TODO: momentum is good in general, but I'm not sure the effect in this context. In general it speeds training and helps escape local minima.
		_qNets[i].SetMomentum(0.1); // algorithm has worked fine w/ and w/out momentum; no perf results; use 0.5ish
	}

	//init the state history; only t and t+1 for now
	_stateHistory.resize(2);
	for(int i = 0; i < _stateHistory.size(); i++){
		_stateHistory[i].resize(NUM_ACTIONS); //each neural net has its own state vector at each time step
		for(int j = 0; j < _stateHistory[i].size(); j++){
			_stateHistory[i][j].resize(STATE_DIMENSION,0.0);
		}
	}

	//set up the historian
	_historyFilePath = "./history.txt";
	_outputFile.open(_historyFilePath,ios::out);
	if(!_outputFile.is_open()){
		cout << "ERROR could not open history.txt file" << endl;
	}
	//write the headers. not using csv for now, oh well...
	//_outputFile << "<comma-delimited state values>,qTarget,qEstimate,ActionString" << endl;

	/*
	//set up the regularization parameters
	for(int i = 0; i < _qNets.size(); i++){	
		_qNets[i].SetRegularizerLambda(0.1);
	}
	*/

}

QAgent::~QAgent()
{
	if(_outputFile.is_open()){
		_outputFile.close();
	}
}

const vector<double>& QAgent::_getPreviousState(Action action)
{
	if(_t == 0)
		return _stateHistory[ _stateHistory.size() - 1 ][(int)action];

	return _stateHistory[_t-1][(int)action];
}

const vector<double>& QAgent::_getCurrentState(Action action)
{
	return _stateHistory[ _t ][(int)action];
}

void QAgent::_updateCurrentState(const World* world, const vector<Missile>& missiles)
{
	//update the time step; this is just an index for _stateHistory for now
	_t = (_t + 1) % (int)_stateHistory.size();
	
	//each neural net has its own state vector, so init the first one, then copy it to the rest
	_deriveCurrentState(world, missiles, _stateHistory[_t][0]);

	//experimental: normalize the state vector such that the neural net inputs are effectively zero-mean
	//_normalizeStateVector(world);
	//_zeroMeanStateVector(world);
}

/*
Normalizes the state vector.

This is experimental.

TODO: Is this desirable over zero-mean? The only reason I'm keeping this around is for testing.
*/
void QAgent::_normalizeStateVector(const World* world, vector<double>& state)
{
	//note cosine attribute is not normalized, since cosine is inherently normalized
	//_stateHistory[_t][SA_XVELOCITY] /= MAX_VELOCITY;
	//_stateHistory[_t][SA_YVELOCITY] /= MAX_VELOCITY;
	state[SA_COLLISION_PROXIMITY] /= MAX_COLLISION_PROXIMITY_RANGE;
	state[SA_GOAL_DIST] /= world->MaxDistanceToGoal;
}

/*
Normalizes all inputs and then zeroes their mean, such that all inputs range [-1.0,1.0].

This is experimental.
*/
void QAgent::_zeroMeanStateVector(const World* world, vector<double>& state)
{
	//note cosine attribute is not normalized, since cosine is inherently normalized
	//_stateHistory[_t][SA_XVELOCITY] /= MAX_VELOCITY;
	//_stateHistory[_t][SA_YVELOCITY] /= MAX_VELOCITY;
	//some magic algebra here. just normalizes and shifts the distance attribute to lie in [-1.0,1.0]
	state[SA_COLLISION_PROXIMITY] =	2.0 * state[SA_COLLISION_PROXIMITY] / MAX_COLLISION_PROXIMITY_RANGE - 1.0;
	state[SA_GOAL_DIST] = 2.0 * state[SA_GOAL_DIST] / world->MaxDistanceToGoal - 1.0;
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
Action QAgent::_getStochasticOptimalAction()
{
	//return _selectionMethod1();
	return _selectionMethod2();
}

//This method uses a more uniform probabilistic selection than method 1,
//which is heavily biased by the most action due to exp().
//TODO: This is incredibly slow for what it accomplishes
Action QAgent::_selectionMethod2()
{
	int i;
	double minQval, r, cdf;
	Action selectedAction;
	vector<double> temp;

	temp.resize(_qNets.size());

	//copy the q-net vals and get the min element
	for(i = 0, minQval = 100000; i < _qNets.size(); i++){
		temp[i] = _qNets[i].GetOutputs()[0].Output;
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

Action QAgent::_selectionMethod1()
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

	/*
	cout << "normalized q vals: ";
	for(i = 0; i < temp.size(); i++){
		cout << temp[i] << " ";
	}
	cout << endl;
	*/

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

//Normalizes a vector of values, consequently making them probabilities, such that their sum is 1.0.
//This is not algebraic unitization.
void QAgent::_normalizeVector(vector<double>& vec)
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
void QAgent::_deriveCurrentState(const World* world, const vector<Missile>& missiles, vector<double>& state)
{
	//tell the agent if it is headed in a good direction: vector-cosine gives a nice value in range [+1.0,-1.0]
	double xHeading, yHeading, x_prime, y_prime;

	//set all the action-unique state values (the state estimate, given the action) for current time _t
	for(int action = 0; action < _stateHistory[_t].size(); action++){
	
		//get the velocity/heading specific state values
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
		//set the velocity values
		//_stateHistory[_t][action][SA_XVELOCITY] = xHeading;
		//_stateHistory[_t][action][SA_YVELOCITY] = yHeading;
		//set distance per the position delta if this action is taken
		_stateHistory[_t][action][SA_GOAL_DIST] = _dist(world->GOAL_X, world->GOAL_Y, agent.x + xHeading, agent.y + yHeading);

		//set the distance to nearest object state attribute per the heading given by each ACTION
		_stateHistory[_t][action][SA_COLLISION_PROXIMITY] = _nearestObjectOnHeading(xHeading, yHeading, world, missiles);

		//set the cossim value for each action net
		//IMPORTANT: If agent is on the goal, zero vector is passed to cossim, which is undefined. In this case,
		//clamp cosine to 1.0 (positive/good) for learning consistency with the goal reward.
		x_prime = world->GOAL_X - (agent.x + xHeading);
		y_prime = world->GOAL_Y - (agent.y + yHeading);
		if(x_prime == 0 && y_prime == 0){ //check to avert passing zero vector to cossim when agent is on the goal
			_stateHistory[_t][action][SA_GOAL_COSINE] = 1.0;
		}
		else{
			_stateHistory[_t][action][SA_GOAL_COSINE] = _cosSim(x_prime, y_prime, xHeading, yHeading);
		}
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
Find's nearest object on given heading, within a radar cone of 45*. If in IDLE, distance is infinite/max.

The solution here is just for the agent to simulate its current velocity for several time steps to estimate
all the locations it will need to traverse.

TODO: a more efficient way to do this. Could query all points up to a distance of d, such that their cosine sim to current heading
is in some range.

@headingX: x component of some heading
@headingY: y component of some heading
Rest are self-explanatory. Distances are relative to agent's current position.
*/
double QAgent::_nearestObjectOnHeading(double headingX, double headingY, const World* world, const vector<Missile>& missiles)
{
	double dist;
	int x_t, y_t;

	//simulate next t time steps at current velocities/heading
	for(int t = 0; t < 15; t++){
		x_t = (int)(headingX * (double)t + agent.x);
		y_t = (int)(headingY * (double)t + agent.y);
		//check if position is either an obstacle of off-map, which is the same as an impenetrable obstacle
		if(!world->IsValidPosition(x_t, y_t) || world->GetCell(x_t, y_t).isObstacle){
			dist = _dist(agent.x, agent.y, (double)x_t, (double)y_t);
			if(dist <= MAX_COLLISION_PROXIMITY_RANGE){
				return dist;
			}
		}
	}

	return MAX_COLLISION_PROXIMITY_RANGE;
}

double QAgent::_dist(double x1, double y1, double x2, double y2)
{
	//points are the same, return zero and avert call to sqrt(0)
	if(x1 == x2 && y1 == y2){
		return 0.0;
	}

	return sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) );
}

/*
Experimental: let's the agent make continous actions (velocities) based on the q values.
*/
void QAgent::_takeContinuousAction()
{
	double ycomp = _qNets[(int)ACTION_UP].GetOutputs()[0].Output - _qNets[(int)ACTION_DOWN].GetOutputs()[0].Output;
	double xcomp = _qNets[(int)ACTION_RIGHT].GetOutputs()[0].Output - _qNets[(int)ACTION_LEFT].GetOutputs()[0].Output;
	double ss = xcomp*xcomp + ycomp*ycomp;
	double normal = 0;
	if(ss > 0){
		//the above check averts a call to sqrt(0)
		normal = sqrt( xcomp*xcomp + ycomp*ycomp );
	}

	/*normalize all the q values
	for(int i = 0; i < _qNets.size(); i++){
		normal += (_qNets[i].GetOutputs()[0].Output * _qNets[i].GetOutputs()[0].Output);
	}
	normal = sqrt(normal);
	*/

	agent.xVelocity = 2 * xcomp / normal;
	agent.yVelocity = 2 * ycomp / normal;
}

/*
Defines the attribute, what is the likelihood of hitting an enemy if we fire from
current position?
*/
double QAgent::_getStrikeLikelihood(const vector<Missile>& missiles)
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
double QAgent::_getMissileLikelihood(const vector<Missile>& missiles)
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


//A flat distance to nearest object, w/out respect to direction, just gives preference
//for the agent to stay away from obstacles.
double QAgent::_nearestObstacleDist(const World* world)
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

//copies a vector to another; user beware on size checking
void QAgent::_copyVec(vector<double>& v1, vector<double>& v2)
{
	for(int i = 0; i < v1.size(); i++){
		v2[i] = v1[i];
	}
}

//Based on the agent's last action, determines if action result in a wall/boundary collision.
bool QAgent::_isWallCollision(const World* world)
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
Get the externally-determined reward value for the agent's current state.
Currently includes:
	-bonks into walls (negative)
	-found goal (positive)
	-hit by missle (negative)

TODO: This likely belongs in the World, such that the World punishes/rewards the agent externally.

Notes: How the reward is characterized has a large effect on the agent's behavior. Do not expect the
network to detect the relative weight of different reward attributes (distance, direction, etc)! I'm
not sure the quality they must have, but in continuous states the goal-reward is not magically back propagated
to previous states, as for discrete models. The rewards must be more instantaneous, and to the point, must
be scaled to roughly equal ranges. For instance, if very close to the goal, then direction should have more weight.
Think about such dependencies. The point here is that the reward function, especially in continuous spaces,
is worthy of study, and how it behaves directly affects the stability of the algorithm.

Also note that when experimenting, certain state attributes can be essentially shut off by setting their associated reward to zero.
*/
double QAgent::_getCurrentRewardValue(const World* world, const vector<Missile>& missiles)
{
	int x_prime, y_prime;
	double reward = 0.0;
	//TODO: better to define rewards as positive/negative or all negative? all negative seems more consistent.
	//double MISSILE_DAMAGE_COST = -10.0;
	double COLLISION_COST = 0.0;
	//double REPETITION_COST = -2.0;
	double GOAL_REWARD = 0.0;
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
double QAgent::_normalizedCosSim(double x1, double y1, double x2, double y2)
{
	double normal1 = sqrt(x1*x1 + y1*y1);
	double normal2 = sqrt(x2*x2 + y2*y2);

	if(normal1 == 0 || normal2 == 0){
		return 0.0;
	}

	return _cosSim(x1/normal1, y1/normal1, x2/normal2, y2/normal2);
}

//cosine similarity for 2d vectors; range neatly fits between -1 and 1.0
double QAgent::_cosSim(double x1, double y1, double x2, double y2)
{
	double cossim;
	double dot = x1*x2 + y1*y2;
	double ss1 = x1*x1 + y1*y1;
	double ss2 = x2*x2 + y2*y2;

	//cout << "args: " << x1 << " " << y1 << " " << x2 << " " << y2 << endl;

	//TODO: This is undefined. What should be returned if a zero-vector is passed? Are there other entry points for the result in the conditional?
	//avert div-zero if either sum-of-squares is zero.
	if(ss1 == 0.0 || ss2 == 0.0 || dot == 0.0){
		cossim = 0.0;
	}
	else{
		cossim = dot / (sqrt(ss1) * sqrt(ss2));
	}
	//cout << "cossim: " << cossim << endl;

	return cossim;
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

double QAgent::_getTargetQValue(const vector<vector<WorldCell> >& world, const vector<Missile>& missiles)
{
	double qTarget = 0.0;

	qTarget = _getCurrentRewardValue(world, missiles) + _gamma * _getMaxQValue();

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
void QAgent::_recordExample(const vector<double>& state, double qTarget, double qEstimate, Action action)
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
void QAgent::ResetEpoch()
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
}

/*

*/
void QAgent::LoopedUpdate(const World* world, const vector<Missile>& missiles)
{
	bool convergence = false;
	int i, iterations;
	double netError = 0.0, maxQ = 0.0, tempMax = 0.0, lastMax = 0.0, qTarget = 0.0;
	Action lastOptimalAction = ACTION_DOWN, optimalAction = ACTION_LEFT, tempAction = ACTION_RIGHT;

	//Update agent's current state and state history
	_updateCurrentState(world, missiles);

	//loop over the q values, retraining to result in some degree of convergence, at least for this action
	maxQ = -999999999;
	lastMax = -999999999;
	for(iterations = 0, convergence = false; iterations < 100 && !convergence; iterations++){
		//classify the new current-state across all action-nets 
		for(i = 0, tempMax = -1000000; i < _qNets.size(); i++){
			//classify the state we just entered, given the previous action
			_qNets[i].Classify(_getCurrentState((Action)i));
			//cout << "classified: " << _qNets[i].GetOutputs()[0].Output << endl;
			//track the max action available in current state
			if(_qNets[i].GetOutputs()[0].Output > tempMax){
				tempMax = _qNets[i].GetOutputs()[0].Output;
				tempAction = (Action)i;
			}
		}
		lastMax = maxQ;
		lastOptimalAction = optimalAction;
		optimalAction = tempAction;
		maxQ = tempMax;

		//detect convergence: if estimate is within 0.1 of previous estimate (verifying also that this is consistent for the same action)
		netError = _absDiff(lastMax,maxQ);
		//cout << "maxq " << maxQ << "  lastMax " << lastMax << "  netError " << netError << endl;
		convergence = (lastOptimalAction == optimalAction) && (netError < 0.05);
		if(!convergence){
			//get the target q factor from the experienced reward given the last action
			qTarget = _getCurrentRewardValue(world, missiles) + _gamma * maxQ;
			//cout << "QTARGET: " << qTarget << endl;
			//backpropagate the error and update the network weights for the last action (only)
			_qNets[(int)CurrentAction].Classify(_getPreviousState((Action)CurrentAction)); //the net must be re-clamped to the previous state inputs and signals
			_qNets[(int)CurrentAction].BackpropagateError(_getPreviousState((Action)CurrentAction), qTarget);
			_qNets[(int)CurrentAction].UpdateWeights(_getPreviousState((Action)CurrentAction), qTarget);
		}
	}

	_epochReward += qTarget;

	//record this example
	_recordExample(_getPreviousState((Action)CurrentAction), qTarget, _qNets[(int)CurrentAction].GetOutputs()[0].Output, CurrentAction);

	//take the action with the highest q-value
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
	if(_episodeCount % 100 == 1){
		for(i = 0; i < _qNets.size(); i++){
			cout << "Action net " << GetActionStr((Action)i) << endl;
			_qNets[i].PrintWeights();
		}
	}
}


/*
TODO: Who acts upon whom? The agent upon the world, or vice versa? There are some theoretical and
code-factoring questions here. Theoretically, there is always an implicit definition of how the agent
realizes its 'target' q value given its last action: by deriving it itself, or by having an unmerciful
world 'tell' it its reward capriciously (mwahaha).

The inputs to this function are the world percepts: the entire world-state and missile states.

The agent then takes some action, experiences a reward, and updates its q-estimates based on
the error in what its q-estimate was for the previous action versus the q-value it experienced.

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
void QAgent::Update(const World* world, const vector<Missile>& missiles)
{
	int i;
	double maxQ, qTarget;
	Action optimalAction = ACTION_UP;

	//Update agent's current state and state history
	_updateCurrentState(world, missiles);

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

		/*
		//experimental: diminishing momentum starts at 0.5, then decreases to 0.0
		if(_totalEpisodes < 2000){
			_qNets[i].SetMomentum( (2000 - _totalEpisodes) / 4000 );
		}
		*/
	}

	//get the target q factor from the experienced reward given the last action
	qTarget = _getCurrentRewardValue(world, missiles) + _gamma * maxQ;
	//cout << "QTARGET: " << qTarget << endl;
	_epochReward += qTarget;
	//cout << "qTarget: " << qTarget << " maxQ: " << maxQ << endl;

	//An edge-case: When the agent just started or when it finds the goal and teleports randomly, don't learn and thereby correlate those unrelated states.
	//if(_episodeCount == 0){
	//for testing: to see if the agent has actually learned, and is not simply following the reward function, cease learning and observe its performance
	/*
	if(_episodeCount == 0 || _epochCount > 50){
		CurrentAction = optimalAction;
		agent.sufferedCollision = false;
		_episodeCount++;
		_totalEpisodes++;
		_takeAction(CurrentAction);
		return;
	}
	*/

	//cout << "currentaction " << (int)CurrentAction << " qnets.size()=" << _qNets.size() << endl;
	//backpropagate the error and update the network weights for the last action (only)
	_qNets[(int)CurrentAction].Classify(_getPreviousState((Action)CurrentAction)); //the net must be re-clamped to the previous state inputs and signals
	//cout << "prev estimate: " << _qNets[(int)CurrentAction].GetOutputs()[0].Output << endl;
	_qNets[(int)CurrentAction].BackpropagateError(_getPreviousState((Action)CurrentAction), qTarget);
	_qNets[(int)CurrentAction].UpdateWeights(_getPreviousState((Action)CurrentAction), qTarget);
	//cout << "44" << endl;

	if(_episodeCount > 10){
		//record this example
		_recordExample(_getPreviousState((Action)CurrentAction), qTarget, _qNets[(int)CurrentAction].GetOutputs()[0].Output, CurrentAction);
	}
	
	//_takeContinuousAction();

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
	_totalEpisodes++;
	/*
	if(_episodeCount % 4000 == 3999){ //random restart every 4000 episodes
		agent.x = 5;
		agent.y = 5;
	}
	*/
	/*
	//detect frozen agent
	if(_getPreviousState()[SA_XPOS] == _getCurrentState()[SA_XPOS] && _getPreviousState()[SA_YPOS] == _getCurrentState()[SA_YPOS]){
		_repetitionCounter++;
		//restart agent if in same place more then 10 time steps
		if(_repetitionCounter > 120){
			_repetitionCounter = 0;
			agent.x = 5 + rand() % 30;
			agent.y = 0 + rand() % 10;
		}
	}
	else{
		_repetitionCounter = 0;
	}
	*/


	//testing: print the neural net weights
	if(_episodeCount % 100 == 1){
		for(i = 0; i < _qNets.size(); i++){
			cout << "Action net " << GetActionStr((Action)i) << endl;
			_qNets[i].PrintWeights();
		}
	}
}

//gets the ordinal distance between two doubles
double QAgent::_absDiff(double d1, double d2)
{
	double d = d1 - d2;

	if(d < 0){
		return -d;
	}
	return d;
}

/*
Learn q-values only when terminal states are reached: collisions, goals, death, etc.


*/

void QAgent::EpochalUpdate(const World* world, const vector<Missile>& missiles)
{
	int i;
	double maxQ, qTarget;
	Action optimalAction = ACTION_UP;

	//Update agent's current state and state history
	_updateCurrentState(world, missiles);

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
	qTarget = _getCurrentRewardValue(world, missiles) + _gamma * maxQ;
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

/*
For this experiment, only update the agent's q-networks when the agent reaches a terminal 
state, or when the error term is very large.

Result: Failure. There may still be potential in something like this, a heuristic for performing
updates only when significant events occur. But this implementation was too divergent. 
*/
void QAgent::DiscriminativeUpdate(const World* world, const vector<Missile>& missiles)
{
	int i;
	double maxQ, qTarget;
	Action optimalAction = ACTION_UP;

	//Update agent's current state and state history
	_updateCurrentState(world, missiles);

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
	qTarget = _getCurrentRewardValue(world, missiles) + _gamma * maxQ;
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

*/
void QAgent::OfflineUpdate(const World* world, const vector<Missile>& missiles)
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
		_updateCurrentState(world, missiles);

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
		qTarget = _getCurrentRewardValue(world, missiles) + _gamma * maxQ;
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


/*
The same implementation as vanilla Update() above, but the agent pushes each (state,qEstimate,qTarget) to
a small, short-term batch of experiences. It then stochastically removes a random experience to train on.
*/
void QAgent::MinibatchUpdate(const World* world, const vector<Missile>& missiles)
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
	_updateCurrentState(world, missiles);

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
	_batch[_batchIndex].QTarget = _getCurrentRewardValue(world, missiles) + _gamma * maxQ;
	_epochReward += _batch[_batchIndex].QTarget;
	//save the action that brought us here
	_batch[_batchIndex].PerformedAction = CurrentAction;

	//update batch index to a new random location in which to store the next experience
	_batchIndex = rand() % _batch.size();
	Experience& exp = _batch[_batchIndex];
	_qNets[(int)exp.PerformedAction].Classify(exp.BatchedState); //the net must be re-clamped to the previous state inputs and signals
	_qNets[(int)exp.PerformedAction].BackpropagateError(exp.BatchedState, exp.QTarget);
	_qNets[(int)exp.PerformedAction].UpdateWeights(exp.BatchedState, exp.QTarget);
	
	/*
	//train over a collection from the batch
	for(i = 0; i < 50; i++){
		Experience& exp = _batch[ rand() % _batch.size() ];
		_qNets[(int)exp.PerformedAction].Classify(exp.BatchedState); //the net must be re-clamped to the previous state inputs and signals
		_qNets[(int)exp.PerformedAction].BackpropagateError(exp.BatchedState, exp.QTarget);
		_qNets[(int)exp.PerformedAction].UpdateWeights(exp.BatchedState, exp.QTarget);
	}
	*/

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
	/*
	if(_episodeCount % 4000 == 3999){ //random restart every 4000 episodes
		agent.x = 5;
		agent.y = 5;
	}
	*/
	/*
	//detect frozen agent
	if(_getPreviousState()[SA_XPOS] == _getCurrentState()[SA_XPOS] && _getPreviousState()[SA_YPOS] == _getCurrentState()[SA_YPOS]){
		_repetitionCounter++;
		//restart agent if in same place more then 10 time steps
		if(_repetitionCounter > 120){
			_repetitionCounter = 0;
			agent.x = 5 + rand() % 30;
			agent.y = 0 + rand() % 10;
		}
	}
	else{
		_repetitionCounter = 0;
	}
	*/


	
	//testing: print the neural net weights
	if(_episodeCount % 100 == 1){
		for(i = 0; i < _qNets.size(); i++){
			cout << "Action net " << GetActionStr((Action)i) << endl;
			_qNets[i].PrintWeights();
		}
	}
	

}





void QAgent::_takeAction(Action nextAction)
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

const char* QAgent::GetActionStr(int i)
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

void QAgent::PrintState()
{
	const vector<double>& s = _getCurrentState(CurrentAction);
	cout << _totalEpisodes << " Epoch " << _epochCount << " Episode " <<  _episodeCount << " Agent (" << agent.x << "x," << agent.y << "y) " << " <xVel yVel cos obstDist goalDist> ";
	//cout << s[SA_XPOS] << " " << s[SA_YPOS] << " " << s[SA_XVELOCITY] << " ";
	cout << agent.xVelocity << " " << agent.yVelocity << " " << s[SA_GOAL_COSINE] << " "  << s[SA_COLLISION_PROXIMITY] << " " << s[SA_GOAL_DIST] << endl;
	//print the epoch measures; ideally all should decrease with training, and avg reward should increase
	cout << "Last epoch performance:  collision rate: " << ((double)((int)(_lastEpochCollisionRate * 1000)) / 100) << "%  #actions: " << _lastEpochActionCount;
	cout << " reward: " << _lastEpochReward << endl;

	cout << "Action (just executed): " << GetActionStr(CurrentAction) << endl;
	for(int i = 0; i < _qNets.size(); i++){
		//_qNets[i].PrintWeights();
		cout << "Output  " << GetActionStr(i) << ": " << _qNets[i].GetOutputs()[0].Output << "  ";
	}

	cout << endl;
}

/*

*/
void QAgent::Train()
{
	
	
	





}




