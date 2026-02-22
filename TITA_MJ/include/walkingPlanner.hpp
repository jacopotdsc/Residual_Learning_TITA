#pragma once

#include <iostream>
#include <fstream>

#include <Eigen/Dense>
#include <map>
#include <string>



namespace labrob {

class walkingPlanner {
  private:
    static constexpr int NX = 13; 
    static constexpr int NU = 9; 
    static constexpr int T = 6;              // in sec
    
    double dt_;
    int N_STEP_;     //n. of timesteps
    
    double grav = 9.81;
    double m = 27.68978;
    
    // ref trajectory
    Eigen::MatrixXd x_ref;
    Eigen::MatrixXd u_ref;

    double vz;
    double v_contact_z;
    double v;
    double omega;
    double theta0;

    double x0;
    double y0;
    double z0;
    double z0_contact;
    double z_min;
    double z_max;


    bool log_plan_ = true;
    
  public: 

  walkingPlanner(const double& dt = 0.002, double vel_lin = 0.0, double vel_ang = 0.0, double vel_z = 0.0, double z0 = 0.4, double z_min = 0.25, double z_max = 0.49) {
    
    std::cout << "Creating WalkingPlanner with dt=" << dt << ", vel_lin=" << vel_lin << ", vel_ang=" << vel_ang << ", vel_z=" << vel_z 
              << ", z0=" << z0 << ", z_min=" << z_min << ", z_max=" << z_max << std::endl;
    dt_ = dt;
    N_STEP_ = static_cast<int>(T / dt_);     //n. of timesteps

    std::cout << "Initializing reference trajectories with " << N_STEP_ << " steps." << std::endl;
    x_ref.setZero(NX, N_STEP_);
    std::cout << "State reference trajectory initialized." << std::endl;
    u_ref.setZero(NU, N_STEP_-1);
    std::cout << "Reference trajectories initialized." << std::endl;

    // fast speed trajectory
    // double T1 = 1;
    // double T2 = 2;
    // double T3 = 3;
    // double T4 = 4;
    // double T5 = 5;
    // double T6 = 7;
    // double T7 = 9;


    
    // constant velocity profile
    vz          = vel_z;
    v_contact_z = 0.0;
    v           = vel_lin;
    omega       = vel_ang;
    theta0      = 0.0;
    x0          = 0.0;
    y0          = 0.0;
    z0          = z0;
    z0_contact  = 0.0;

    z_min       = z_min;
    z_max       = z_max;
    std::cout << "WalkingPlanner created with " << N_STEP_ << " steps." << std::endl;
  }
  
  void offline_plan() {
    double x = x0;
    double y = y0;
    double z = z0;

    for (int t_step = 0; t_step < N_STEP_; ++t_step){
        double t = t_step * dt_;


        // if (t < T1){
        //   v = 1.1;
        // } else if (t >= T1  && t < T2){
        //   v = 2.5;
        // } else if (t >= T2 && t < T3){
        //   v = 4.0;
        // } else if (t >= T3  && t < T4){
        //   v = 5.0;
        // } else if (t >= T4 && t < T5){
        //   v = 4.0;
        // } else if (t >= T5 && t < T6){
        //   v = 3.0;
        // } else if (t >= T6 && t < T7){
        //   v = 2.0;
        // } else if (t >= T7){
        //   v = 1.0;
        // }


        // unicycle equations
        double theta = theta0 + omega * t;
        double vx = v * cos(theta);
        double vy = v * sin(theta);

        // Euler integration
        if (abs(omega) > 1e-9) {      // assume v always constant along the trajectory and integrate from p0
            x = x0 + (v / omega) * (sin(theta0 + omega * t) - sin(theta0));
            y = x0 - (v / omega) * (cos(theta0 + omega * t) - cos(theta0));
        } else {                      // assume v piece-wise constant along the trajectory and compute the increment from previous position
            x  += v * cos(theta0) * dt_;
            y  += v * sin(theta0) * dt_;
        }

        z = std::clamp(z + vz * dt_, z_min, z_max);
        double z_contact = z0_contact + v_contact_z * t;

        if (z <= z_min || z >= z_max){
          vz = 0.0;
        }


        // stop at last state
        if (t_step == N_STEP_ - 1){
          vx = 0.0; vy = 0.0; vz = 0.0;
          v_contact_z = 0.0; v = 0.0; omega = 0.0;
        }

        x_ref.col(t_step)(0)  = x;
        x_ref.col(t_step)(1)  = y;
        x_ref.col(t_step)(2)  = z;

        x_ref.col(t_step)(3)  = vx;
        x_ref.col(t_step)(4)  = vy;
        x_ref.col(t_step)(5)  = vz;

        x_ref.col(t_step)(6)  = x;
        x_ref.col(t_step)(7)  = y;
        x_ref.col(t_step)(8)  = z_contact;

        x_ref.col(t_step)(9)  = v_contact_z;
        x_ref.col(t_step)(10) = theta;
        x_ref.col(t_step)(11) = v;
        x_ref.col(t_step)(12) = omega;  
    }



    for (int t_step = 0; t_step < N_STEP_ - 1; ++t_step){

        u_ref.col(t_step)(0) = 0.0;     // a
        u_ref.col(t_step)(1) = 0.0;     // ac_z
        u_ref.col(t_step)(2) = 0.0;     // alpha
        
        u_ref.col(t_step)(3) = 0;         // fl_x
        u_ref.col(t_step)(4) = 0;         // fl_y
        u_ref.col(t_step)(5) = m*grav/2;  // fl_z

        u_ref.col(t_step)(6) = 0;         // fr_x
        u_ref.col(t_step)(7) = 0;         // fr_y
        u_ref.col(t_step)(8) = m*grav/2;  // fr_z
    }


    if (log_plan_){
      // create folder if it does not exist
      std::string folder = "/tmp/plan/" ;
      std::string command = "mkdir -p " + folder;
      const int ret = std::system(command.c_str());
      (void)ret;

      // print trajectory to file
      std::string path_x = "/tmp/plan/x.txt";
      std::ofstream file_x(path_x);
      for (int i = 0; i < N_STEP_; ++i) {
        file_x << x_ref.col(i).transpose() << std::endl;
      }
      file_x.close();
      std::string path_u = "/tmp/plan/u.txt";
      std::ofstream file_u(path_u);
      for (int i = 0; i < N_STEP_ - 1; ++i) {
        file_u << u_ref.col(i).transpose() << std::endl;
      }
      file_u.close();
    }
  }

  void jumpRoutine(const double& t_msec, const double h_jump){

    double t0 = t_msec / 1000;
    int current_time_step = get_time_step_idx(t_msec);
    double com_z_cur = x_ref.col(current_time_step)(2);

    double T_down = 0.3;
    double T_up = 0.3;

    double t_in = t0 + T_down;
    double g = 9.81;
    double v0_jump = std::sqrt(2 * g * h_jump);

    double T_jump = 2 * v0_jump / g;
    
    double T_total = T_jump + T_down + T_up;
    if(t0 + T_total > T){
      stop_trajectory(t_msec);
      return;
    }
    int N_STEP_JUMP = static_cast<int>(T_total / dt_) + 1;


    double vz          = 0.0;
    double v_contact_z = 0.0;
  
    double z0_contact  = 0.0;

    // TODO: chek if z_jump goes under z_min
    // double z_min       = 0.2;

    double z_start_jump = 0.3;

    double z = com_z_cur;
    double z_contact = z0_contact;

    // down cubic poly params
    double v0_down = vz * T_down;
    double vf_down = v0_jump * T_down;

    double a_down = 2 * z - 2 * z_start_jump + v0_down + vf_down;
    double b_down = 3 * z_start_jump - 3 * z - 2 * v0_down - vf_down;
    double c_down = v0_down;
    double d_down = z;

    // down cubic poly params
    double v0_up = (v0_jump - g * T_jump) * T_up;
    double vf_up = vz * T_up;

    double a_up = 2 * z_start_jump - 2 * z + v0_up + vf_up;
    double b_up = 3 * z - 3 * z_start_jump - 2 * v0_up - vf_up;
    double c_up = v0_up;
    double d_up = z_start_jump;

    for (int t_step = 0; t_step < N_STEP_JUMP; ++t_step){
        double t = t0 + t_step * dt_;
         
        if (t < t0 + T_down){
          double tau = (t - t0) / T_down;
          tau = std::clamp(tau, 0.0, 1.0);
          z = a_down * tau * tau * tau + b_down * tau * tau + c_down * tau + d_down;
          vz = (3 * a_down * tau * tau + 2 * b_down * tau + c_down) * 1 / T_down;
          z_contact = 0.0;
          v_contact_z = 0.0;
        } else if (t >= t0 + T_down  && t < t0 + T_down + T_jump + (dt_)){  // jump time   dt_ introduced because the integration is discrete and at time t it could be still in landing 
          double tj = t - t_in;
          z  = z_start_jump + v0_jump * tj - 0.5 * g * tj * tj;
          vz = v0_jump - g * tj;
          z_contact = 0.0 + v0_jump * tj - 0.5 * g * tj * tj;
          z_contact = std::max(z_contact, 0.0);
          v_contact_z = vz;
        } else if (t >= t0 + T_down + T_jump + (dt_)){
          double tau = (t - (t0 + T_down + T_jump)) / T_up;
          tau = std::clamp(tau, 0.0, 1.0);
          z = a_up * tau * tau * tau + b_up * tau * tau + c_up * tau + d_up;
          vz = (3 * a_up * tau * tau + 2 * b_up * tau + c_up) * 1 / T_up;
          z_contact = 0.0;
          v_contact_z = 0.0;
        }

        x_ref.col(current_time_step + t_step)(2)  = z;
        x_ref.col(current_time_step + t_step)(5)  = vz;

        x_ref.col(current_time_step + t_step)(8)  = z_contact;

        x_ref.col(current_time_step + t_step)(9)  = v_contact_z;    
    }


    if (log_plan_){
      // print trajectory to file
      std::string path_jump = "/tmp/plan/jump_traj.txt";
      std::ofstream file_jump(path_jump);
      file_jump << t_msec << " index " << current_time_step << std::endl;
      for (int i = 0; i < N_STEP_JUMP; ++i) {
        file_jump << x_ref.col(current_time_step + i).transpose() << std::endl;
      }
      file_jump.close();
    }

  }



  void stop_trajectory(const double& t_msec){

    double t0 = t_msec / 1000;
    int current_time_step = get_time_step_idx(t_msec);

    int N_CURR = static_cast<int>(t0 / dt_);

    double com_x_cur = x_ref.col(current_time_step)(0);
    double com_y_cur = x_ref.col(current_time_step)(1);
    double com_z_cur = x_ref.col(current_time_step)(2);
    double theta_cur = x_ref.col(current_time_step)(10);
    for (int t_step = N_CURR; t_step < N_STEP_; ++t_step){
        double t = t0 + t_step * dt_;

        x_ref.col(t_step)(0)  = com_x_cur;
        x_ref.col(t_step)(1)  = com_y_cur;
        x_ref.col(t_step)(2)  = com_z_cur;

        x_ref.col(t_step)(3)  = 0.0;
        x_ref.col(t_step)(4)  = 0.0;
        x_ref.col(t_step)(5)  = 0.0;

        x_ref.col(t_step)(6)  = com_x_cur;
        x_ref.col(t_step)(7)  = com_y_cur;
        x_ref.col(t_step)(8)  = 0.0;

        x_ref.col(t_step)(9)  = 0.0;
        x_ref.col(t_step)(10) = theta_cur;
        x_ref.col(t_step)(11) = 0.0;
        x_ref.col(t_step)(12) = 0.0;  
    }
  }


  int get_time_step_idx(const double& t_msec) const{
   return static_cast<int>(std::llround(t_msec / 1000 / dt_));         // check the rounding when control timestep is not fixed
  }

  Eigen::MatrixXd get_xref_at_time_ms(const double& t_msec) const {
    int time_step = get_time_step_idx(t_msec);
    time_step = std::clamp(time_step, 0, N_STEP_ - 1);
    return x_ref.col(time_step);
  }
  Eigen::MatrixXd get_uref_at_time_ms(const double& t_msec) const {
    int time_step = get_time_step_idx(t_msec);
    time_step = std::clamp(time_step, 0, N_STEP_ - 2);
    return u_ref.col(time_step);
  }


  const Eigen::MatrixXd& get_x_ref() const {
    return x_ref;
  }

  const Eigen::MatrixXd& get_u_ref() const {
      return u_ref;
  }

  std::map<std::string, double> getVariables() const {
    std::map<std::string, double> vars;

    vars["NX"] = static_cast<double>(NX);
    vars["NU"] = static_cast<double>(NU);
    vars["T"] = static_cast<double>(T);
    vars["dt"] = dt_;
    vars["N_STEP"] = static_cast<double>(N_STEP_);
    vars["grav"] = grav;
    vars["m"] = m;
    
    vars["dim_x_ref_rows"] = static_cast<double>(x_ref.rows());
    vars["dim_x_ref_cols"] = static_cast<double>(x_ref.cols());
    vars["dim_u_ref_rows"] = static_cast<double>(u_ref.rows());
    vars["dim_u_ref_cols"] = static_cast<double>(u_ref.cols());
    
    vars["vz"] = vz;
    vars["v_contact_z"] = v_contact_z;
    vars["v"] = v;
    vars["omega"] = omega;
    vars["theta0"] = theta0;
    vars["x0"] = x0;
    vars["y0"] = y0;
    vars["z0"] = z0;
    vars["z0_contact"] = z0_contact;
    vars["z_min"] = z_min;
    vars["z_max"] = z_max;

    return vars;
  }

}; 

} // end namespace labrob
