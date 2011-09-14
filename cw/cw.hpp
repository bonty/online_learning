#ifndef _CW_HPP__
#define _CW_HPP__

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

namespace cw_tool {
  typedef std::vector<std::pair<int, double> > FV;
  typedef std::vector<double> VD;

  int trainData(const char* trainFile, const char* modelFile,
                const double conf, const int iter);

  int testData(const char* testFile, const char* modelFile);

  class cw {
  public:
    cw();
    ~cw();

    int trainData(const char* filename, const int iter = 10);
    int testData(const char* filename, std::vector<int>& result);

    int saveModel(const char* modelFile);
    int loadModel(const char* modelFile);

    void setConf(const double conf);

    std::string getErrorLog() const;

  private:
    double conf;                // confidence parameter
    VD     w;                   // weight vector
    VD     cov;                 // covariance

    std::ostringstream errorLog;

    int parseLine(const std::string& line, FV& fv, int& label);
    
    void trainExample(const FV& fv, const int label);

    double getMargin(const FV& fv) const;
    double getVariance(const FV& fv) const;

    void update(const FV& fv, const int label, const double alpha);

    template<class T>
    void valWrite(const T& v, std::ofstream* fp, const char* name);

    template<class T>
    void vecWrite(const std::vector<T>& v, std::ofstream* fp, const char* name);

    template<class T>
    void valRead(T&, std::ifstream* fp, const char* name);

    template<class T>
    void vecRead(std::vector<T>& v, std::ifstream* fp, const char* name);
    
  };

  // Template methods

  template<class T>
  void cw::valWrite(const T& v, std::ofstream* fp, const char* name) {
    T val = v;
    (*fp).write(reinterpret_cast<char*>(&val), sizeof(T));
  }

  template<class T>
  void cw::vecWrite(const std::vector<T>& v, std::ofstream* fp, const char* name) {
    size_t n = v.size();
    valWrite(n, fp, name);
    for (size_t i = 0; i < n; ++i) {
      T val = v[i];
      valWrite(val, fp, name);
    }
  }
  
  template<class T>
  void cw::valRead(T& v, std::ifstream* fp, const char* name) {
    (*fp).read(reinterpret_cast<char*>(&v), sizeof(T));
  }
  
  template<class T>
  void cw::vecRead(std::vector<T>& v, std::ifstream* fp, const char* name) {
    size_t n = 0;
    valRead(n, fp, name);
    v.clear();
    v.resize(n);

    for (size_t i = 0; i < n; ++i) {
      T val;
      valRead(val, fp, name);
      v[i] = val;
    }
  }
}

#endif // _CW_HPP__
