#ifndef POLICY_H_
#define POLICY_H_

class Policy {

public:
  virtual void run(int cycle) = 0;
  virtual ~Policy() {};
};

#endif
