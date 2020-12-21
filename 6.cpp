#include <iostream>
using namespace std;
void circularshift(int& i,int& j,int& k)
{
	int temp1,temp2,temp3;
	temp1=i;
	temp2=j;
	temp3=k;
	j=temp1;
	k=temp2;
	i=temp3;
	
}
	
	
//StudybarCommentBegin

int main( )
{ 
     int i,j,k;
	 cin >> i >>j >>k;
     circularshift(i,j,k);
     cout<<i<<" "<<j<<" "<<k<<endl;
     return 0;
}
//StudybarCommentEnd
