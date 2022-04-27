#include "parser.h"

int Parser::stringparam2int(std::string& arg)
{
    for (auto pa : long_ljlparam)
    {
        //arg.erase(std::remove(arg.begin(), arg.end(), '-'), arg.end());
        //std::cout << "update arg" << arg << std::endl;
        if (arg == pa.name or arg == pa.val)
        {
            return pa.flag;
        }
    }
    return -1;
}


std::vector<std::string> Parser::split(const std::string& str, const std::string& pattern)
{
	char* strc = new char[strlen(str.c_str()) + 1];
	strcpy(strc, str.c_str());   //string×ª»»³ÉC-string
	std::vector<std::string> res;
	char* temp = strtok(strc, pattern.c_str());
	while (temp != NULL)
	{
		res.push_back(std::string(temp));
		temp = strtok(NULL, pattern.c_str());
	}
	delete[] strc;
	return res;
}

bool Parser::paraseArgv(int argc, char** argv, SampleParams& params)
{

	if (argc == 1)
	{
		std::cout << "jin lai" << std::endl;
		return true;
	}
	std::cout << "jin lai" << std::endl;
	//std::cout << "long_options->name[0]" << long_ljlparam[0].name<< std::endl;
	//std::cout << "long_options->name[0]" << long_ljlparam->name[1] << std::endl;
	for (int i = 1; i < argc; i++)
	{
		int flag;
		std::vector<std::string> res;
		std::string mid = argv[i];
		res = split(mid, "=");
		if (res.size() != 2)
		{
			std::cout << "command line parsing failed  " << std::endl;
			std::cout << "It is possible that the equals sign is missing " << std::endl;
			return false;
		}

		std::string env = res[0];
		flag = stringparam2int(env);
		if (flag == -1)
		{
			std::cout << "command not found£º" << mid << std::endl;
			return false;
		}
		switch (flag)
		{
		case 0:
			params.imgRpath = res[1]; break;
		case 1:
			params.imgWpath = res[1]; break;
		case 2:
			params.videoRurl = res[1]; params.videoRFlag = true; break;
		case 3:
			params.videoWurl = res[1]; params.videoSFlag = true;  break;
		case 4:
			params.loadEngine = res[1]; break;
		case 5:
			params.saveEngine = res[1]; params.savef = true; break;
		case 6:
			params.int8 = true; break;
		case 7:
			params.fp16 = true; break;
		case 8:
			params.batchSize = 1; break;
		case 9:
			params.dlaCore = 1; break;
		case 10:
			params.loadOnnx = res[1]; break;
		case 11:
			params.savef = true; break;

		default:
			break;
		}

	}
	return true;
}
