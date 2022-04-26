#pragma once
#include<iostream>
#include"baseStruct.h"

struct ljlparam /* specification for a long form option...	*/
{
	const char* name; /* option name, without leading hyphens */
	int flag;        /* where to save its status, or NULL	*/
	const char* val;          /* its associated status value		*/

};

static struct ljlparam long_ljlparam[] = {
	{"imgReadPath", 0, "-ir"},{"imgWritePath", 1, "-iw"}, {"videoReadUrl", 2, "-vr"}, {"videoWriteUrl", 3, "-vw"},
	{"loadEnginePath", 4, "-le"},{"saveEnginPath",5, "-se"},{"int8", 6, "-i8"}, {"fp16", 7, "-f16"},
	{"batch", 8, "-b"},{"useDLACore",  9, "-d"},{"loadOnnxPath", 10, "-lo"}, {"saveFlag", 11, "-sf"}
};


class Parser
{
public:


	int stringparam2int(std::string& arg);
	std::vector<std::string> split(const std::string& str, const std::string& pattern);
	bool paraseArgv(int argc, char** argv, SampleParams& params);

private:
	
};