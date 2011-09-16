#include <cmath>
#include <algorithm>

#include "cw.hpp"

namespace cw_tool {
  void vdDump (const VD& v) {
    for (size_t i = 0; i < v.size(); ++i) {
      std::cout << i << ':' << v[i] << ' ';
    }
    std::cout << std::endl;
  }
  int trainData(const char* trainFile, const char* modelFile,
                const double conf, const int iter) {
    cw cw;
    cw.setConf(conf);

    bool error = false;
    if (cw.trainData(trainFile, iter) == -1)
      error = true;

    if (error) {
      std::cerr << cw.getErrorLog() << std::endl;
      return -1;
    }
    else {
      if (cw.saveModel(modelFile) == -1) {
        std::cerr << cw.getErrorLog() << std::endl;
        return -1;
      }
    }

    return 0;
  }

  int testData(const char* testFile, const char* modelFile) {
    cw cw;
    if (cw.loadModel(modelFile) == -1) {
      std::cerr << cw.getErrorLog() << std::endl;
      return -1;
    }

    std::vector<int> result;
    if (cw.testData(testFile, result) == -1) {
      std::cerr << cw.getErrorLog() << std::endl;
      return -1;
    }

    const int correct = result[0] + result[3];
    const int sum = result[0] + result[1] + result[2] + result[3];
    printf("accuracy %.3f%% (%d/%d)\n"
           "(Answer, Predict): (p,p):%d (p,n):%d (n,p):%d (n,n):%d\n",
           100.f * correct / sum, correct, sum,
           result[0], result[1], result[2], result[3]);
    
    return 0;
  }

  cw::cw() : conf(1.f) {};
  cw::~cw() {};

  int cw::trainData(const char* filename, const int iter) {
    if (conf <= 0) {
      errorLog << "confidence parameter is less than zero:" << conf;
      return -1;
    }
    if (iter <= 0) {
      errorLog << "iteration number is less than zero:" << iter;
      return -1;
    }
    
    std::ifstream ifs(filename);
    if (!ifs) {
      errorLog << "cannot open " << filename;
      return -1;
    }

    size_t lineNum = 0;
    std::string line;
    std::vector<FV> sfv; // sample feature vector
    std::vector<int> labels;

    while(getline(ifs, line)) {
      ++lineNum;
      if (line[0] == '#') continue; // comment line

      FV fv;
      int label = 0;
      if (parseLine(line, fv, label) == -1) {
        errorLog << "line:" << lineNum;
        return -1;
      }

      sfv.push_back(fv);
      labels.push_back(label);
    }

    ifs.close();

    std::cout << "Read done." << std::endl;

    // std::random_shuffle(sfv.begin(), sfv.end());

    for (int i = 0; i < iter; ++i) {
      for (size_t j = 0; j < sfv.size(); j++) {
        trainExample(sfv[j], labels[j]);
      }

      if (iter > 10) {
        std::cout << ".";
        if ((iter+1) % 50 == 0)
          std::cout << std::endl;
      }
    }

    std::cout << "Finish!" << std::endl;

    return 0;
  }

  int cw::testData(const char* filename, std::vector<int>& result) {
    std::ifstream ifs(filename);
    if (!ifs) {
      errorLog << "cannot open " << filename;
      return -1;
    }

    size_t lineNum = 0;
    std::string line;

    result.clear();
    result.resize(4, 0);

    while (getline(ifs, line)) {
      ++lineNum;
      if (line[0] == '#') continue;

      FV fv;
      int label = 0;
      if (parseLine(line, fv, label) == -1) {
        errorLog << "line:" << lineNum;
        return -1;
      }

      const double score = getMargin(fv);

      if (score >= 0 && label == 1) { // pp
        ++result[0];
      }
      else if (score < 0 && label == 1) { // pn
        ++result[1];
      }
      else if (score >= 0 && label == -1) { // np
        ++result[2];
      }
      else if (score < 0 && label == -1) { // nn
        ++result[3];
      }
      else {
        errorLog << "error score:" << score << " label:" << label << " line:" << lineNum;
        return -1;
      }
    }

    return 0;
  }

  int cw::saveModel(const char* filename) {
    std::ofstream ofs(filename, std::ofstream::binary);
    if (!ofs) {
      errorLog << "Unable to open " << filename;
      return -1;
    }

    valWrite(conf, &ofs, "conf");
    vecWrite(w, &ofs, "w");
    vecWrite(cov, &ofs, "cov");

    ofs.close();
    
    return 0;
  }

  int cw::loadModel(const char* filename) {
    std::ifstream ifs(filename, std::ifstream::binary);
    if (!ifs) {
      errorLog << "Unable to open " << filename;
      return -1;
    }

    valRead(conf, &ifs, "conf");
    vecRead(w, &ifs, "w");
    vecRead(cov, &ifs, "cov");

    ifs.close();

    return 0;
  }

  void cw::setConf(const double conf_) {
    conf = conf_;
  }

  std::string cw::getErrorLog() const {
    return errorLog.str();
  }
  
  int cw::parseLine(const std::string& line, FV& fv, int& label) {
    std::istringstream is(line);
    if (!(is >> label)) {
      errorLog << "parse error: no label ";
      return -1;
    }

    if (label != 1 && label != -1) {
      errorLog << "parse error: label is not +1 nor -1 ";
    }

    int id = 0;
    char delim = 0;
    double val = 0.f;

    while (is >> id >> delim >> val) {
      fv.push_back(std::make_pair(id, val));
    }
    
    return 0;
  }
  
  void cw::trainExample(const FV& fv, const int label) {
    double margin = getMargin(fv) * label;
    double variance = getVariance(fv);

    double b = 1.f + 2.f * conf * margin;
    double gamma = (-b + sqrt(b*b - 8.f * conf * (margin - conf * variance))) /
      (4.f * conf * variance);

    if (gamma > 0) {
      update(fv, label, gamma);
    }
  }

  double cw::getMargin(const FV& fv) const {
    double ret = 0.f;
    for (size_t i = 0; i < fv.size(); ++i) {
      if (w.size() <= (size_t)fv[i].first)
        continue;
      ret += w[fv[i].first] * fv[i].second;
    }
    return ret;
  }

  double cw::getVariance(const FV& fv) const {
    double ret = 0.f;
    for (size_t i = 0; i < fv.size(); ++i) {
      if(cov.size() <= (size_t)fv[i].first) {
        ret += 1.f * fv[i].second + fv[i].second; // initial value of cov[i] is 1.0
      }
      else {
        ret += cov[fv[i].first] * fv[i].second * fv[i].second;
      }
    }
    return ret;
  }

  void cw::update(const FV& fv, const int label, const double alpha) {
    for (size_t i = 0; i < fv.size(); ++i) {
      if (cov.size() <= (size_t)fv[i].first) {
        w.resize(fv[i].first+1, 0.f);
        cov.resize(fv[i].first+1, 1.f);
      }

      w[fv[i].first] += alpha * label * cov[fv[i].first] * fv[i].second;
      cov[fv[i].first] = 1.f /
        (1.f/cov[fv[i].first] + 2.f * alpha * conf * fv[i].second * fv[i].second);
    }
  }

}
