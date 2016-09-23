#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <iostream>
#include <errno.h>
#include <string.h>

using namespace std;


int main(int argc, char** argv, char** env)
{
	int pid, status;
	char* pyScript = "./History2Datasets.py";
	char* const args[] = {"History2Datasets.py history.txt", NULL};

	//fire a python proc to split the log into data for each network
	if(pid = fork()){ //parent waits for python proc to complete
		//wait for child python proc to return, meaning the per-action training data has been written
		waitpid(pid,&status,0);
		cout << "wait returned" << endl;
	}
	else{
		//execv("./hello",args);
		//execve("./hello",args,env);
		execlp("python", "python", "History2Datasets.py", "history.txt", (char*) NULL);
		//execve("/usr/bin/python",args, env);
		cout << "should never see this line" << endl;
		cout << "error: " << strerror(errno) << endl;
	}

	return 0;
}
