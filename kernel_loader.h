#pragma once
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

class KernelReader {
private:
	string source;
public:
	KernelReader(const char* filename);
	const char* getKernelSource() { return source.c_str(); };
};