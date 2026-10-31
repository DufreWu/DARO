#ifndef INPUTARGS_H_
#define INPUTARGS_H_

#include "argparse.hpp"


struct MyArgs : public argparse::Args {
    int &policy_id = kwarg("p,policy", "id of the controller policy");    
    int &init_freq_b = kwarg("fb,init-freq-big", "initial frequency of big cluster").set_default(499200);
    int &init_freq_l = kwarg("fl,init-freq-little", "initial frequency of little cluster").set_default(499200);
    int &init_freq_g = kwarg("fg,init-freq-gpu", "initial frequency of gpu cluster").set_default(216750000);
    float &init_speed = kwarg("s,init-speed", "initial speed").set_default(2.5);
    std::string &log_dir = kwarg("log-dir", "path to dir where csv log is written to");
    std::string &model_dir = kwarg("model-dir", "path to dir where model parameters are stored").set_default("../../chen_resource_controller/models");
    bool &verbose         = flag("v,verbose", "A flag to toggle verbose");
};

#endif
