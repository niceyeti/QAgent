#include "Q2Agent.hpp"


int main(int argc, char** argv, char** env)
{
	string dummy;
	World world(120, 35);
	Q2Agent q2Agent(5,5);
	vector<Missile> missiles;

	srand(time(NULL));
	usleep(rand() % 50000);
	world.InitializeWorld(missiles, q2Agent.agent);

	while(true){

		//tell the agent the current world state, letting it take some action
		//qAgent.OfflineUpdate(&world,missiles);
		//qAgent.MinibatchUpdate(&world,missiles);
		//qAgent.DiscriminativeUpdate(&world,missiles);
		q2Agent.LoopedUpdate(&world,missiles);
		//qAgent.EpochalUpdate(&world,missiles);
		//q2Agent.Update(&world,missiles);

		//Update and draw the world; this is just updating a model and displaying a view
		world.Update(missiles, &q2Agent, 1.0);
		world.Draw(missiles, q2Agent.agent);
		q2Agent.PrintState();
		//usleep(50000);
	}

	return 0;
}






