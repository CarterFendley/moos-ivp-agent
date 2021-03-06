/************************************************************/
/*    NAME: Carter Fendley                                              */
/*    ORGN: MIT                                             */
/*    FILE: BHV_Agent.h                                      */
/*    DATE: 2021/07/22                                      */
/************************************************************/

#ifndef Agent_HEADER
#define Agent_HEADER

#include <string>
#include "IvPBehavior.h"
#include "PyInterface.h"

class BHV_Agent : public IvPBehavior {
public:
  BHV_Agent(IvPDomain);
  ~BHV_Agent() {};

  bool         setParam(std::string, std::string);
  void         onSetParamComplete();
  void         onCompleteState();
  void         onIdleState();
  void         onHelmStart();
  void         postConfigStatus();
  void         onRunToIdleState();
  void         onIdleToRunState();
  IvPFunction* onRunState();

protected: // Local Utility functions
  void         postBridgeState(std::string state);
  void         tickBridge();

protected: // Configuration parameters
  std::vector<std::string> m_sub_vars;
  std::vector<std::string> m_sub_vehicles;

protected: // State variables
  PyInterface bridge;
  double m_current_course;
  double m_current_speed;
};

#define IVP_EXPORT_FUNCTION

extern "C" {
  IVP_EXPORT_FUNCTION IvPBehavior * createBehavior(std::string name, IvPDomain domain)
  {return new BHV_Agent(domain);}
}
#endif
