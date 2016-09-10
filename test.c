#include <cstdio>
#include <unistd.h>

/*
PLaying with terminal colors. Use 'reset' to return terminal to normal.

This code only works for whatever terminals support printf control codes.
See wiki on ANSII control codes for more.
*/


int main(void)
{
	int i, j;
	char* rainbow = "RAINBOW";

	printf("a line\n");
	printf("another line\n");
	usleep(1000000);


	/*
	Most terminals support ANSI escape codes. You can use a J (with parameter 2)
	to clear the screen and an H (with parameters 1,1) to reset the cursor to the top-left:
	printf("\033[2J\033[1;1H");
	*/
	printf("\033[2J\033[1;1H"); //clear the terminal and return cursor to top-left
	

	//green letters, white background: \033[32
	//printf("%c[%dmHELLO!\n", 0x1B, 32);
	//printf("\033[32 HELLO!\n");

	//printf("%c[%dmHELLO!\n", 0x1B, 47);  // "ESC[30;47m"

	//printf("\033[30;47m"); 	//black letters, white background
	//printf("\033[31;47m");  //red letters, white background
	printf("\033[34;47m");  //blue letters, white background
	//printf("\033[32;47m");  //green letters, white background
	//and so on, where red=1, green=2, yell=3, blue=4, magenta=5, cyan=6, white=7

	
	//alternation is possible
	printf("\033[34;47m BLUE! ");  //blue letters, white background
	usleep(1000000);
	printf("\033[32;47m GREEN!\n");  //green letters, white background
	usleep(1000000);
	//a bloody friggin rainbow!
	for(i = 0; i < 7; i++){
		printf("\033[3%d;47m",i);  //blue letters, white background
		printf("%c",rainbow[i]);	
	}
	printf("\n");


	printf("fresh lijnes\n");
	printf("ahgdsghgf\n");
	//usleep(2000000);

	//usleep(500000);
	//printf("\033[2J\033[1;1H"); //clear the terminal and return cursor to top-left
	printf("Type 'reset' in your terminal to return to normal\n");

	return 0;
}
