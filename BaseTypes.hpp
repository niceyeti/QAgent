#ifndef BASE_TYPES_HPP
#define BASE_TYPES_HPP

#include "../ANN/MultilayerNetwork.hpp"
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

#define GOAL_Y 32
#define GOAL_X 100
//The actions. Each of which will have its own neural network
#define NUM_ACTIONS 5
//enum Action{ACTION_UP,ACTION_DOWN,ACTION_LEFT,ACTION_RIGHT,ACTION_SHOOT,ACTION_IDLE};
enum Action{ACTION_UP,ACTION_DOWN,ACTION_LEFT,ACTION_RIGHT,ACTION_IDLE};
//The agent's state-dimension, and the number of inputs to each q-net
#define STATE_DIMENSION 6 //  velocity x-component, velocity y-component, x position, y position, etc in spec
enum StateAttribute{SA_XPOS, SA_YPOS, SA_XVELOCITY, SA_YVELOCITY, SA_OBSTACLE_DIST, SA_GOAL_DIST};

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
	double xVelocity;
	double yVelocity;
	//double xAcceleration;
	//double yAcceleration;
}Agent;

#endif
