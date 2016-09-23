#include <iostream>

using namespace std;

/*
Just wanted to see if ansi-control codes work with cout as they do with printf.
*/


int main(void)
{

	cout << "a line\n" << endl;
	cout << "another line\n" << endl;
	cout << "\033[2J\033[1;1H" << endl; //clear the terminal and return cursor to top-left
	

	//green letters, white background: \033[32
	//printf("%c[%dmHELLO!\n", 0x1B, 32);
	//printf("\033[32 HELLO!\n");

	//printf("%c[%dmHELLO!\n", 0x1B, 47);  // "ESC[30;47m"

	//printf("\033[30;47m"); 	//black letters, white background
	//printf("\033[31;47m");  //red letters, white background
	cout << "\033[34;47m BLUE!!" << endl;  //blue letters, white background
	//printf("\033[32;47m");  //green letters, white background
	//and so on, where red=1, green=2, yell=3, blue=4, magenta=5, cyan=6, white=7

	return 0;
}
