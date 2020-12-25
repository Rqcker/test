#include <iostream>
using namespace std;
int fac(int);  
int fac(int x) 
{  
    int f;  
  
    if(x==0 || x==1)  
        f=1;  
    else  
        f=fac(x-1)*x;  
  
    return f;  
}   
int main()  
{  
    int n;  
  
    while(cin>>n)  
    {  
        cout<<fac(n)<<endl;  
    }  
  
    return 0;  
}  
  

