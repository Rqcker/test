#include <stdio.h>
void swap(int a, int b)
{
    int t;
    t=a;
    a=b;
    b=t;
}

main(int argc, char ** argv)
{
    int x,y;
    x=1;
    y=2;
    swap(x,y);
    printf("%d,%d",x,y);
}
