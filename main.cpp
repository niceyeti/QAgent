#include "QAgent.hpp"


int main(int argc, char** argv, char** env)
{
	string dummy;
	World world(120, 35);
	QAgent qAgent(5,5);
	vector<Missile> missiles;

	srand(time(NULL));
	usleep(rand() % 50000);
	world.InitializeWorld(missiles, qAgent.agent);

	while(true){
		//tell the agent the current world state, letting it take some action
		//qAgent.OfflineUpdate(&world,missiles);
		//qAgent.MinibatchUpdate(&world,missiles);
		//qAgent.DiscriminativeUpdate(&world,missiles);

		for(int i = 0; i < 20; i++){
			qAgent.Update(&world,missiles);
		}
		//cin >> dummy;

		//qAgent.EpochalUpdate(&world,missiles);

		//Update and draw the world; this is just updating a model and displaying a view
		world.UpdateWorld(missiles, qAgent, 1.0);
		world.DrawWorld(missiles, qAgent.agent);
		qAgent.PrintState();
		//usleep(50000);
	}

	return 0;
}






