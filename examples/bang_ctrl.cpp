#include <iostream>
#include <unistd.h>
#include <string>
#include <fstream>
#include "unitree_arm_sdk/control/unitreeArm.h"

using namespace std;

int main(int argc, char **argv)
{
    Eigen::IOFormat CleanFmt(4, Eigen::DontAlignCols, ",", "\n", "", "");
    
    int j;
    if (argc < 2) {
        cout << "No joint specified. Using joint 0" << endl;
        j = 0;
    }
    else {
        cout<< "Using joint " << argv[1] << endl;
        j = stoi(argv[1]);
    }

    if (j < 0 || j > 5) {
        cout << "Invalid joint number. Must be between 0 and 5" << endl;
        return -1;
    }

    // 1) Initialize robot
    UNITREE_ARM::unitreeArm arm(true);
    arm.sendRecvThread->start();
    arm.backToStart();
    arm.startTrack(UNITREE_ARM::ArmFSMState::JOINTCTRL);

    // Configuration
    double dt = arm._ctrlComp->dt;
    double n_pre = 1000;
    int bang_cycles = 3;
    int T_bang = 500;
    int n_cycle = T_bang * 4;
    double a_bang = 1;
    double joint_delta = a_bang * pow(T_bang * dt, 2);
    cout << "Joint delta: " << joint_delta << " rad" << endl;

    Vec6 q_init, q_arm_up, q_per_joint;
    q_init = arm.lowstate->getQ();
    q_arm_up << 0., 0.26178, -0.26178, 0., 0., 0.;
    q_per_joint << 0, 0.05, -1.2, -0.7, 0, 0;
    q_arm_up[j] = q_per_joint[j];

    // Define the acceleration profile
    VecX a_cycle = VecX::Zero(n_cycle);
    a_cycle.head(T_bang).setConstant(a_bang);
    a_cycle.segment(T_bang, 3 * T_bang).setConstant(-a_bang);
    a_cycle.tail(T_bang).setConstant(a_bang);

    // Replicate the acceleration profile for number of cycle
    VecX a = a_cycle.replicate(bang_cycles, 1);

    string filename_log =  "../data/state_log_" + to_string(j) + ".csv";
    ofstream state_log(filename_log);

    // 2) Move arm to initial position and block it
    cout << "[PRE]" << endl;
    UNITREE_ARM::Timer timer(arm._ctrlComp->dt);
    for (int i = 0; i < n_pre; i++) {
        arm.q = q_init * (1 - i / n_pre) + q_arm_up * (i / n_pre);
        arm.qd = (q_arm_up - q_init) / (n_pre * dt);
        arm.setArmCmd(arm.q, arm.qd);
        timer.sleep();
    }
    // cout << " Actual velocity: " << arm.qd.transpose().format(CleanFmt) << endl;
    // cout << " Measured velocity: " << arm.lowstate->getQd().transpose().format(CleanFmt) << endl;

    cout << "[BLOCK]" << endl;
    for (int i = 0; i < 10; i++) {
        arm.jointCtrlCmd(Vec7::Zero(), 0);
    }

    // 3) Bang-bang control
    sleep(2); 
    cout << "[BANG]" << endl;
    double q_target = arm.q[j];

    for (int i = 0; i < a.size(); i++) {
        arm.q[j] += arm.qd[j] * dt + 0.5 * a[i] * pow(dt, 2);
        // q_target += q;
        arm.qd[j] += a[i] * dt;
        arm.setArmCmd(arm.q, arm.qd);

        // Log data
        state_log << arm.lowstate->getQ().transpose().format(CleanFmt) << ',' 
                  << arm.lowstate->getQd().transpose().format(CleanFmt) << ','
                  << arm.lowstate->getTau().transpose().format(CleanFmt) << endl;

        timer.sleep();
    }

    state_log.close();

    // 4) Move arm back to initial position
    cout << "[BACK]" << endl;
    arm.backToStart();
    arm.setFsm(UNITREE_ARM::ArmFSMState::PASSIVE);
    arm.sendRecvThread->shutdown();
    return 0;
}
