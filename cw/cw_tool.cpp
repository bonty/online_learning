#include <iostream>
#include <stdlib.h>

#include "cw.hpp"

void usage() {
  std::cerr << "Usage: tw_tool (train|test) (trainFile|testFile) modelFile [-c <confidence param> -i <iteration>]"
            << std::endl;
}

void parseArg(char* argv[], int& idx, double& conf, int& iter) {
  std::string str(argv[idx]);

  if (str == "-c") {
    conf = atof(argv[++idx]);
  }
  else if (str == "-i") {
    iter = atoi(argv[++idx]);
  }
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    usage();
    return -1;
  }

  double conf = 1.0f;
  int iter = 1;

  for (int i = 4; i < argc; ++i) {
    parseArg(argv, i, conf, iter);
  }

  std::string argv1(argv[1]);
  if (argv1 == "train") {
    if (cw_tool::trainData(argv[2], argv[3], conf, iter) == -1)
      return -1;
  }
  else if (argv1 == "test") {
    if (cw_tool::testData(argv[2], argv[3]) == -1)
      return -1;
  }
  else {
    usage();
    return -1;
  }

  return 0;
}
