//
// Created by arx4060 on 24-12-4.
//

#pragma once

#include "arx_hardware_interface/can_motor_dlc/Mtp3Dlc.hpp"
#include "arx_hardware_interface/canbase/SocketCan.hpp"
#include "arx_hardware_interface/chassis_dlc/Chassis1Dlc.hpp"

using namespace arx;
using namespace hw_interface;

namespace arx {
class LiftHeadControlLoop {
 public:
  enum class RobotType {
    LIFT,
    X7S
  };
  LiftHeadControlLoop(const char *bus_name, RobotType robot_type);
  ~LiftHeadControlLoop();
  void setRobotType(RobotType robot_type);
  void setHeight(double height) { lift_motor_pos_des_ = height * 41.54; }
  void setWaistPos(double pos);
  void setHeadYaw(double yaw_des);
  void setHeadPitch(double pitch_des);
  void setChassisCmd(double v_x, double v_y, double w_z, int mode);
  double getHeight();
  double getWaistPos();
  double getHeadYaw();
  double getHeadPitch();
  void read();
  void update();
  void write();
  void loop() {
    read();
    update();
    write();
  }

 protected:
  double max_height_ = 20;
  double lift_motor_pos_des_{}, waist_motor_pos_des_{}, head_yaw_motor_pos_des_{},
      head_pitch_motor_pos_des_{};
  double lift_pos_current_des_{};
  double waist_pos_current_des_{};
  double lift_motor_torque_des_{}, lift_motor_vel_des_{};
  bool lift_is_calibrated_ = false, waist_is_calibrated_ = false;
  bool head_initial_pos_flag_ = false;
  bool head_init_complete_ = false;
  int head_init_count_ = 0;
  int calibrated_count_ = 0;
  int waist_calibrated_count_ = 0;
  // chassis
  int mode_ = 2;
  int p_x_ = 0, p_y_ = 0, p_z_ = 0;
  double max_vel_x_, max_vel_y_, max_vel_z_;

 private:
  SocketCan socket_can_;
  MotorType3 lift_motor_, waist_motor_, head_yaw_motor_, head_pitch_motor_;
  ChassisType1 chassis_;
  void recieveFrameCallback(CanFrame *frame);
  void readFromBuffer();
};
}// namespace arx
