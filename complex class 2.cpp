#include <iostream>
using namespace std;
class Cmycomplex
{
private:
	int im,re;
public:
	Cmycomplex(int i=0, int r=0);
//	Cmycomplex();
//	Cmycomplex(int i);
    int GetReal();
    int GetImaginary();
	void Show();
	
};
Cmycomplex::Cmycomplex(int i, int r)
{
	im=r;
	re=i;
}
/*Cmycomplex::Cmycomplex()
{
	im=0;
	re=0;
} 
Cmycomplex::Cmycomplex(int i)
{
	im=i;
	re=0;
}*/ 

void Cmycomplex::Show()
{
	if(im>=0)
	cout <<"("<< re << "+" << im <<"i)";
	else
	cout <<"("<< re << "-" << -im <<"i)";

};
int Cmycomplex::GetReal()
{
	return re;
}
int Cmycomplex::GetImaginary()
{
	return im;
}
//StudybarCommentBegin
int main()
{
Cmycomplex  z1(2,3),z2,z3(3);
cout<<z1.GetReal()<<"\n";
cout<<z2.GetImaginary()<<"\n";
cout<<z3.GetReal()<<"\n";

}
//StudybarCommentEnd
