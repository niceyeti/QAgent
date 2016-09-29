#ifndef Q2UTIL_HPP
#define Q2UTIL_HPP

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <unistd.h>
#include <cmath>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <cstdio>

#define _USE_MATH_DEFINES
#ifndef M_PI
	#define M_PI 3.141592653589793238462643383279
#endif



/*
An implementation of approximate q-learning for discrete state/action spaces,
in this case just a grid-world. This differs from tabular q-learning in that
a neural network is used to estimate Q-values instead of a huge table. This uses 
a discrete state/action representation as a personal tutorial, and extension to
continuous state spaces should be trivial once the discrete-world version works.

Neural nets for learning q-values are entirely defined by the user; some have 
a separate neural net for each action, and all generally have different architectures.
In other words, I have no idea which is best, this is just an attempted approximate
q-learning implementation for a very simple environment, as a self tutorial.

The architecture and update method is directly from Haykin, Neural Networks: A Comprehensive...
*/

using namespace std;

//The actions. Each of which will have its own neural network
#define NUM_ACTIONS 4
//enum Action{ACTION_UP,ACTION_DOWN,ACTION_LEFT,ACTION_RIGHT,ACTION_SHOOT,ACTION_IDLE};
enum Action{ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT};
//The agent's state-dimension, and the number of inputs to each q-net
#define STATE_DIMENSION 3 //  velocity x-component, velocity y-component, x position, y position, etc in spec
//enum StateAttribute{SA_XPOS, SA_YPOS, SA_XVELOCITY, SA_YVELOCITY, SA_COLLISION_PROXIMITY, SA_GOAL_DIST};
//enum StateAttribute{SA_XVELOCITY, SA_YVELOCITY, SA_GOAL_COSINE, SA_COLLISION_PROXIMITY, SA_GOAL_DIST};
//enum StateAttribute{SA_GOAL_COSINE, SA_RECENT_LOCATION_COSINE, SA_COLLISION_PROXIMITY, SA_GOAL_DIST};
enum StateAttribute{SA_GOAL_COSINE, SA_RECENT_LOCATION_COSINE, SA_COLLISION_PROXIMITY};
//!!!!! WARNING!!!! Change any of the above, and verify the entire group. Eg, don't add a state attribute and not STATE_DIMENSION too!!!!

#define BATCH_SIZE 200

//these are required to normalize the state vector, such that the neural net inputs are range-constrained
//MAX_COSINE isn't needed, since it is inherently normalized; and MAX_GOAL_DIST is dependent on the world
#define MAX_VELOCITY 1.0
//if this is too large, the agent can become too conservative, not finding narrow passages between/around objects,
//in short, smaller sensitivity gives greater precision in the agent's ability to avoid objects
#define MAX_COLLISION_PROXIMITY_RANGE 6.0

typedef struct missile{
	double acceleration;
	double theta;
	double velocity;
	double x;
	double y;
}Missile;

//Trivial data type describing a particular grid location
typedef struct worldCell{
	bool isObstacle;
	bool isTraversed;
	bool isGoal;
}WorldCell;

typedef struct agent{
	double x;
	double y;
	bool isAlive;
	bool sufferedCollision; //terrible coupling through this member
	double xVelocity;
	double yVelocity;
	//double xAcceleration;
	//double yAcceleration;
}Agent;

#endif
