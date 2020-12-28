#include <iostream>
using namespace std;
class Myclock
{
private:
	int hour, min, sec;
public:
	void Set(int h, int m, int s)
	{
		hour = h;
		min = m;
		sec = s;
	};
	void Show()
	{
		cout << hour << ":" << min << ":" << sec;
	};
};
int main()
{
	int a;
	double b;
	char c;
	Myclock clock1;
	clock1.Set(23, 25, 38);
	clock1.Show();
}