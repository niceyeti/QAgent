#include "World.hpp"

World::World()
{
	GOAL_X = 100;
	GOAL_Y = 30;
	_worldString = NULL;
	ResizeWorld(120, 35);
	_setMaxGoalDist();
}

World::World(int x, int y)
{
	GOAL_X = (x * 4) / 5;
	GOAL_Y = (y * 4) / 5;
	_worldString = NULL;
	ResizeWorld(x, y);
	_setMaxGoalDist();
}

World::~World()
{
	if(_worldString != NULL){
		delete _worldString;
	}
}

int World::MaxX() const
{
	return _maxX;
}

int World::MaxY() const
{
	return _maxY;
}

/*
//just for debugging
void World::PrintState(vector<Missile>& missiles, DiscreteQ2Agent& qagent)
{
	cout << "world rows: " << _world.size() << "  cols: " << _world[0].size() << endl;
	cout << "agent position: " << qagent->agent.x << "," << qagent->agent.y << endl;
	for(int i = 0; i < missiles.size(); i++){
		cout << "m" << i << ": " << missiles[i].x << "," << missiles[i].y << endl;
	}
}
*/

//Wrapper for converting x/y coordinates with origin at bottom-left to matrix coordinates for the world matrix.
WorldCell& World::GetCell(int x, int y)
{
	//TODO: validate indices. I'd prefer crash in prototype code though.
	return _world[ _world.size() - y - 1 ][x];
}

const WorldCell& World::GetCell(int x, int y) const
{
	//TODO: validate indices. I'd prefer crash in prototype code though.
	return _world[ _world.size() - y - 1 ][x];
}

void World::ResizeWorld(int x, int y)
{
	_maxY = y - 1;
	_maxX = x - 1;

	_world.resize(y);
	for(int i = 0; i < _world.size(); i++){
		_world[i].resize(x);
	}
}

void World::InitializeWorld(vector<Missile>& missiles, Agent& agent)
{
	int i, j, x, y;

	for(i = 0; i < _world.size(); i++){
		for(j = 0; j < _world[i].size(); j++){
			_world[i][j].isObstacle = false;
			_world[i][j].isTraversed = false;
			_world[i][j].isGoal = false;
		}
	}

	//place some random obstacles
	for(i = 0; i < 30; i++){
		//get a random orientation, horizontal or vertical
		if((rand() % 2) == 0){ //place a vertical obstacle
			x = rand() % _world[0].size();
			y = rand() % (_world.size() - 14) + 10; //magic math places obstacles lower in view
			//place the obstacle using the start coordinates from the bottom
			for(j = y; j < (y + 4); j++){
				_world[j][x].isObstacle = true;
			}
		}
		else{ //place a horizontal obstacle
			x = rand() % (_world[0].size() - 7);
			y = rand() % (_world.size() - 14) + 10;
			//place the obstacle using the start coordinates from the bottom
			for(j = x; j < (x + 7); j++){
				_world[y][j].isObstacle = true;
			}
		}
	}


	//init the missiles
	//InitRandomMissiles(missiles, 8);

	//place the goal randomly in the upper region
	_setRandomGoalLocation();
}

/*
int max(int x1, int x2)
{
	if(x1 <= x2)
		return x2;
	return x1;
}

int min(int x1, int x2)
{
	if(x1 >= x2)
		return x2;
	return x1;
}
*/
void World::InitRandomMissiles(vector<Missile>& missiles, int numMissiles)
{
	missiles.resize(numMissiles);

	//init numMissiles missiles in random x-columns, travelling straight down
	for(int i = 0; i < missiles.size(); i++){
		//velocity range = 0.5-1.5
		missiles[i].velocity = 1.0 * ((double)(rand() % 100 + 50) / 100.0);
		missiles[i].acceleration = 0.0;
		//theta range: pi/4 - 3pi/4
		//missiles[i].theta = (M_PI / 2) * ((double)(rand() % 100 + 50) / 100.0);
		missiles[i].theta = -(M_PI / 2); //straight down
		missiles[i].x = rand() % (int)_world[0].size();
		missiles[i].y = 0;
	}
}

//Note this 'takes' the agent's action for the agent, based on its velocity.
void World::Update(vector<Missile>& missiles, Q2Agent* qagent, double timeStep)
{
	int newX, newY;

	//upate agent
	//teleport to (0,0) if goal reached
	if(GetCell(qagent->agent.x, qagent->agent.y).isGoal){
		qagent->EpochGoalCount++;
		//cout << "AGENT FOUND GOAL, epoch ct=" << qagent->EpochGoalCount << " thresh=" << qagent->GoalResetThreshold << endl;
		//cin >> newX;
	}

	//allows the agent to find the goal a few times before resetting
	if(qagent->EpochGoalCount >= qagent->GoalResetThreshold){
		_restartAgent(qagent,1.0);
	}
	else{
		newX = qagent->agent.x + qagent->agent.xVelocity * timeStep;	
		newY = qagent->agent.y + qagent->agent.yVelocity * timeStep;
		//inform the agent of any collision
		if(!IsValidPosition(newX,newY) || GetCell(newX,newY).isObstacle){
			qagent->agent.sufferedCollision = true;
			qagent->EpochCollisionCount++;
			qagent->StoreTerminalState(-1.0); //ugly hack: store terminal state w/out resetting the agent
			//_restartAgent(qagent,-1.0);
		}
		//else{

		//the following checks prevent the agent from moving out of bounds or through obstacles
		if(newX > _maxX){
			newX = _maxX;
		}
		else if(newX < 0){
			newX = 0;
		}
		if(newY > _maxY){
			newY = _maxY;
		}
		else if(newY < 0){
			newY = 0;
		}
		if(!GetCell(newX, newY).isObstacle){
			qagent->agent.x = (double)newX;
			qagent->agent.y = (double)newY;
		}
		//}
	}

	//if(IsValidPosition(agent.x, agent.y)){
	GetCell(qagent->agent.x, qagent->agent.y).isTraversed = true;
	//}

	/*
	for(int i = 0; i < missiles.size(); i++){
		missiles[i].x += (missiles[i].velocity * cos(missiles[i].theta) * timeStep);
		missiles[i].y += (missiles[i].velocity * sin(missiles[i].theta) * timeStep);
		missiles[i].velocity += (missiles[i].acceleration * timeStep);
		//wrap the missile at _world edges
		if(missiles[i].x > (double)_maxX){
			missiles[i].x = 0.0;
		}
		else if(missiles[i].x < 0){
			missiles[i].x = _maxX;
		}
		if(missiles[i].y > (double)_maxY){
			missiles[i].y = 0.0;
		}
		else if(missiles[i].y < 0){
			missiles[i].y = _maxY;
		}
	}
	*/
}

//wrapper util for resetting the agent and restarting it in a random start location
void World::_restartAgent(Q2Agent* qagent, double terminalValue)
{
	_setRandomGoalLocation();
	_setRandomAgentStartLocation(qagent->agent);
	_resetTraversalFlags();
	qagent->ResetEpoch(terminalValue);
}


void World::_setMaxGoalDist()
{
	//oriented with the lower left grid as the origin, the max goal distance is always just the dist from origin to goal
	MaxDistanceToGoal = sqrt(MaxX()*MaxX() + MaxY()*MaxY());
	//MaxDistanceToGoal = sqrt(pow((double)GOAL_X - agent.x, 2.0) + pow((double)GOAL_Y - agent.y, 2.0));
}


void World::_setRandomGoalLocation()
{
	//erase existing location
	GetCell(GOAL_X, GOAL_Y).isGoal = false;
	//set new location
	GOAL_X = rand() % _maxX;
	GOAL_Y = rand() % (_maxY / 5) + ((4 * _maxY) / 5);
	GetCell(GOAL_X, GOAL_Y).isGoal = true;
}

//randomly place the agent in the lower portion of the world
void World::_setRandomAgentStartLocation(Agent& agent)
{
	agent.x = rand() % (_maxX / 8);
	agent.y = rand() % (_maxY / 6);
}

//On teleportation back to start, all isTraversed flags are reset.
void World::_resetTraversalFlags()
{
	int i, j;

	for(i = 0; i < _world.size(); i++){
		for(j = 0; j < _world[i].size(); j++){
			_world[i][j].isTraversed = false;
		}
	}
}

//Check if a position is in bounds.
bool World::IsValidPosition(int x, int y) const
{
	return (x >= 0 && x <= _maxX) && (y >= 0 && y <= _maxY);
}

void World::Draw(vector<Missile>& missiles, Agent& agent)
{
	int i, j, stringIndex;
	int lineWidth = _world[0].size() + 1; //plus one for the newline of each row of the grid _world
	int len = sizeof(char) * _world.size() * lineWidth + 1; //plus 1 for null char

	if(_worldString == NULL){
		_worldString = new char[len];
	}

	//zero the string
	memset((void*)_worldString, 0, len);

	for(i = 0; i < len; i++){
		if((i % lineWidth) == (lineWidth - 1)){
			_worldString[i] = '\n';
		}
		else{
			_worldString[i] = ' ';
		}
	}
	_worldString[len] = '\0';

	//warn of bad placements
	
                                   
	//draw the obstacles
	for(i = 0; i < _world.size(); i++){
		for(j = 0; j < _world[i].size(); j++){
			if(_world[i][j].isObstacle){
				_worldString[ i * lineWidth + j ] = '#';
			}
		}
	}


	//overlay the agent's path
	for(i = 0; i < _world.size(); i++){
		for(j = 0; j < _world[i].size(); j++){
			if(_world[i][j].isTraversed){
				_worldString[ i * lineWidth + j ] = '-';
			}
		}
	}

	//overlay all the missile positions
	for(i = 0; i < missiles.size(); i++){
		stringIndex = (int)(_world.size() - missiles[i].y - 1) * lineWidth + (int)missiles[i].x;
		_worldString[stringIndex] = 'v';
	}

	//place the Goal
	stringIndex = (int)(_world.size() - GOAL_Y - 1) * lineWidth + GOAL_X;
	_worldString[stringIndex] = 'G';

	//draw the agent's position
	stringIndex = (int)(_world.size() - agent.y - 1) * lineWidth + (int)agent.x;
	_worldString[stringIndex] = 'o';

	//cout << _worldString << endl;
	//printf("%c[2J cleared screen",27); //cls
	//printf("\033[2J\033[1;1H\n"); //clear the terminal and return cursor to top-left
	//printf("%s\n",_worldString);
	//cout << "\033[2J" << endl; //cls; also try \027??
	cout << "\033[2J\033[1;1H" << endl; //clear the terminal and return cursor to top-left
	cout << _worldString << endl;
	cout << "GOAL X/Y: " << GOAL_X << " " << GOAL_Y << endl;
}
