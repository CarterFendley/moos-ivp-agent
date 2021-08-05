/************************************************************/
/*    NAME: Carter Fendley                                              */
/*    ORGN: MIT                                             */
/*    FILE: BHV_Agent.cpp                                    */
/*    DATE: 2021/07/22                                      */
/************************************************************/

#include <iterator>
#include <cstdlib>
#include "MBUtils.h"
#include "BuildUtils.h"
#include "BHV_Agent.h"
#include "VarDataPair.h"
#include "ZAIC_PEAK.h"
#include "OF_Coupler.h"

using namespace std;

//---------------------------------------------------------------
// Constructor

BHV_Agent::BHV_Agent(IvPDomain domain) :
  IvPBehavior(domain)
{
  // Provide a default behavior name
  IvPBehavior::setParam("name", "rl_agent");

  // Declare the behavior decision space
  m_domain = subDomain(m_domain, "course,speed");

  // Add any variables this behavior needs to subscribe for
  addInfoVars("NAV_X, NAV_Y", "NAV_HEADING");
  addInfoVars("NODE_REPORT_LOCAL");
  addInfoVars("EPISODE_MNGR_REPORT");

  if(true)
    setbuf(stdout, NULL);
}

//---------------------------------------------------------------
// Procedure: setParam()

bool BHV_Agent::setParam(string param, string val)
{
  // Convert the parameter to lower case for more general matching
  param = tolower(param);

  // Get the numerical value of the param argument for convenience once
  double double_val = atof(val.c_str());

  if(param == "sub_vehicle") {
    m_sub_vehicles.push_back(toupper(stripBlankEnds(val)));
    return(true);
  }
  else if(param == "sub_var"){
    m_sub_vars.push_back(stripBlankEnds(val));
    return(true);
  }
  else if (param == "bar") {
    // return(setBooleanOnString(m_my_bool, val));
  }

  // If not handled above, then just return false;
  return(false);
}

//---------------------------------------------------------------
// Procedure: onSetParamComplete()
//   Purpose: Invoked once after all parameters have been handled.
//            Good place to ensure all required params have are set.
//            Or any inter-param relationships like a<b.

void BHV_Agent::onSetParamComplete()
{
  unsigned int i, vsize = m_sub_vehicles.size();
  for(i=0; i<vsize; i++){
    addInfoVars("NODE_REPORT_"+m_sub_vehicles[i]);
  }
  vsize = m_sub_vars.size();
  for(i=0; i<vsize; i++){
    addInfoVars(m_sub_vars[i]);
  }
}

//---------------------------------------------------------------
// Procedure: onHelmStart()
//   Purpose: Invoked once upon helm start, even if this behavior
//            is a template and not spawned at startup

void BHV_Agent::onHelmStart()
{
}

//---------------------------------------------------------------
// Procedure: onIdleState()
//   Purpose: Invoked on each helm iteration if conditions not met.

void BHV_Agent::onIdleState()
{
  tickBridge(false);
}

//---------------------------------------------------------------
// Procedure: onCompleteState()

void BHV_Agent::onCompleteState()
{
  tickBridge(false);
}

//---------------------------------------------------------------
// Procedure: postConfigStatus()
//   Purpose: Invoked each time a param is dynamically changed

void BHV_Agent::postConfigStatus()
{
}

//---------------------------------------------------------------
// Procedure: onIdleToRunState()
//   Purpose: Invoked once upon each transition from idle to run state

void BHV_Agent::onIdleToRunState()
{
}

//---------------------------------------------------------------
// Procedure: onRunToIdleState()
//   Purpose: Invoked once upon each transition from run to idle state

void BHV_Agent::onRunToIdleState()
{
}

//---------------------------------------------------------------
// Procedure: onRunState()
//   Purpose: Invoked each iteration when run conditions have been met.

IvPFunction* BHV_Agent::onRunState()
{
  IvPFunction *ipf = 0;
  tickBridge(true);

  // Listen for action from bridge
  std::vector<VarDataPair> action = bridge.listenAction();
  
  int vsize = action.size();
  if(vsize > 0){
    // We got an action
    for(int i=0; i<vsize; i++){
      string var = action[i].get_var();

      if (var == "speed"){
        m_current_speed = action[i].get_ddata();
      }else if (var == "course"){
        m_current_course = action[i].get_ddata();
      }else{
        // The action should be a MOOS_VAR action
        if(action[i].is_string()){
          postRepeatableMessage(var, action[i].get_sdata());
        }else{
          postRepeatableMessage(var, action[i].get_ddata());
        }
      }
    }
  }else{
    // Bridge didn't get an action but failed nicely (timeout)
  }

  // Build a new IvP function
  ZAIC_PEAK spd_zaic(m_domain, "speed");
  spd_zaic.setSummit(m_current_speed);
  spd_zaic.setBaseWidth(0.3);
  spd_zaic.setPeakWidth(0.0);
  spd_zaic.setSummitDelta(0.0);
  IvPFunction *spd_of = spd_zaic.extractIvPFunction();

  ZAIC_PEAK crs_zaic(m_domain, "course");
  crs_zaic.setSummit(m_current_course);
  crs_zaic.setBaseWidth(180.0);
  crs_zaic.setValueWrap(true);
  IvPFunction *crs_of = crs_zaic.extractIvPFunction();

  OF_Coupler coupler;
  ipf = coupler.couple(crs_of, spd_of);

  // Part N: Prior to returning the IvP function, apply the priority wt
  // Actual weight applied may be some value different than the configured
  // m_priority_wt, depending on the behavior author's insite.
  if(ipf)
    ipf->setPWT(m_priority_wt);

  return(ipf);
}

void BHV_Agent::postBridgeState(std::string state){
  postRepeatableMessage("AGENT_BRIDGE_STATE", state);
  postRepeatableMessage("AGENT_CURRENT_ACTION", "speed="+doubleToString(m_current_speed)+",course="+doubleToString(m_current_course));
}

//---------------------------------------------------------------
// Procedure: tickBridge()
//   Purpose: Used to tick the bridge
void BHV_Agent::tickBridge(bool running){
  // Post status if failed
  if(bridge.failureState()){
    postBridgeState("Failed");
    return;
  }

  // Post status if connected
  if(!bridge.isConnected()){
    bridge.connect();
    postBridgeState("Not Connected");
    return; // Nothing else to do
  }
  
  postBridgeState("Connected");
  // Send the current state
  if(running){
    // Pull NAV_X and NAV_Y from the Helm info buffer
    bool x_ok, y_ok, h_ok;
    double NAV_X = getBufferDoubleVal("NAV_X", x_ok);
    if(!x_ok){
      postWMessage("NAV_X not found in info buffer. Can't send state update.");
      return;
    }
    double NAV_Y = getBufferDoubleVal("NAV_Y", y_ok);
    if(!y_ok){
      postWMessage("NAV_Y not found in info buffer. Can't send state update.");
      return;
    }
    double NAV_HEADING = getBufferDoubleVal("NAV_HEADING", h_ok);
    if(!h_ok){
      postWMessage("NAV_HEADING not found in info buffer. Can't send state update.");
      return;
    }

    bool name_ok;
    std::string node_local = getBufferStringVal("NODE_REPORT_LOCAL", name_ok);
    if(!name_ok){
      postWMessage("NODE_REPORT_LOCAL not found in info buffer. Can't sent state update.");
    }
    std::string VNAME = tokStringParse(node_local, "NAME", ',', '=');

    // Post other node reports
    std::vector<std::string> node_reports;
    unsigned int i, vsize = m_sub_vehicles.size();
    for(i=0; i<vsize; i++){
      bool ok;
      std::string result = getBufferStringVal("NODE_REPORT_"+m_sub_vehicles[i], ok);
      if(ok){
        node_reports.push_back(result);
      }
    }

    // Look for vars that are subscribed to
    std::vector<VarDataPair> vd_pairs;
    vsize = m_sub_vars.size();
    for(i=0; i<vsize; i++){
      // Access buffer directly as we don't know what type
      // Calls through Helm will throw un wanted warnings
      bool ok_s, ok_d;
      string s_result = m_info_buffer->sQuery(m_sub_vars[i], ok_s);
      double d_result = m_info_buffer->dQuery(m_sub_vars[i], ok_d);

      if(ok_d){
        VarDataPair pair(m_sub_vars[i], d_result);
        vd_pairs.push_back(pair);
      }else if(ok_s){
        VarDataPair pair(m_sub_vars[i], s_result);
        vd_pairs.push_back(pair);
      }else{
        postWMessage("Subscription var '"+m_sub_vars[i]+"' not found in info buffer");
      }
    }

    // Add pEpisodeManager report or null if not present
    bool report_ok;
    string report = m_info_buffer->sQuery("EPISODE_MNGR_REPORT", report_ok);
    if(!report_ok)
      report = "null";
    VarDataPair pair("EPISODE_MNGR_REPORT", report);
    vd_pairs.push_back(pair);

    // Send update through bridge
    bool ok = bridge.sendState(getBufferCurrTime(), NAV_X, NAV_Y, NAV_HEADING, VNAME, node_reports, vd_pairs);
    if (!ok){
      postWMessage("Bridge says connected but failed to send state.");
    }
  }
}