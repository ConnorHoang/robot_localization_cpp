#include <cstddef>
#define _USE_MATH_DEFINES

#include <cmath>
#include <iostream>
#include <rclcpp/time.hpp>
#include <string>
#include <tuple>
#include <random>

#include <angles/angles.h>

#include "angle_helpers.hpp"
#include "builtin_interfaces/msg/time.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "nav2_msgs/msg/particle_cloud.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "helper_functions.hpp"
#include "occupancy_field.hpp"
#include "pf.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

using std::placeholders::_1;


#include <optional>

Particle::Particle(float w, float theta, float x, float y)
{
  this->w = w;
  this->theta = theta;
  this->x = x;
  this->y = y;
}

/**
 * A helper function to convert a particle to a geometry_msgs/Pose message
 */
geometry_msgs::msg::Pose Particle::as_pose()
{
  geometry_msgs::msg::Pose pose = geometry_msgs::msg::Pose();
  pose.position.x = this->x;
  pose.position.y = this->y;
  pose.orientation = quaternion_from_euler(0, 0, this->theta);

  return pose;
}

ParticleFilter::ParticleFilter() : Node("pf"), uniform_distribution_(0.0f, 1.0f)
{
  base_frame = "base_footprint"; // the frame of the robot base
  map_frame = "map";             // the name of the map coordinate frame
  odom_frame = "odom";           // the name of the odometry coordinate frame
  scan_topic = "scan";           // the topic where we will get laser scans from

  d_thresh = 0.2; // the amount of linear movement before performing an update
  a_thresh = M_PI / 6; // the amount of angular movement before performing an update

  // Declare parameters with their default values
  this->declare_parameter<int>("n_particles", 300); // the number of particles to use

  // resample params
  this->declare_parameter<double>("resampling.truncation_percentage", 0.25);
  this->declare_parameter<double>("resampling.random_percentage", 0.05);
  this->declare_parameter<double>("resampling.noise_x_stddev", 0.1);
  this->declare_parameter<double>("resampling.noise_y_stddev", 0.1);
  this->declare_parameter<double>("resampling.noise_theta_stddev", 0.05 * M_PI);

  // Get the parameters and store them in member variables
  this->n_particles = this->get_parameter("n_particles").as_int();

    // resample params
  this->get_parameter("resampling.truncation_percentage", truncation_percentage_);
  this->get_parameter("resampling.random_percentage", random_percentage_);
  this->get_parameter("resampling.noise_x_stddev", resample_noise_x_stddev_);
  this->get_parameter("resampling.noise_y_stddev", resample_noise_y_stddev_);
  this->get_parameter("resampling.noise_theta_stddev", resample_noise_theta_stddev_);

  // parameter update callback
  auto callback = std::bind(&ParticleFilter::on_parameters_changed, this, std::placeholders::_1);
  param_callback_handle_ = this->add_on_set_parameters_callback(callback);

  // pose_listener responds to selection of a new approximate robot
  // location (for instance using rviz)
  auto sub1_opt = rclcpp::SubscriptionOptions();
  sub1_opt.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  initial_pose_subscriber = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "initialpose", 10,
      std::bind(&ParticleFilter::update_initial_pose, this, _1),
      sub1_opt);

  // publish the current particle cloud.  This enables viewing particles
  // in rviz.
  particle_pub = this->create_publisher<nav2_msgs::msg::ParticleCloud>(
      "particle_cloud", 10);

  auto sub2_opt = rclcpp::SubscriptionOptions();
  sub2_opt.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  // laser_subscriber listens for data from the lidar
  laserscan_subscriber = this->create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic,
      10,
      std::bind(&ParticleFilter::scan_received, this, _1),
      sub2_opt);

  // this is used to keep track of the timestamps coming from bag files
  // knowing this information helps us set the timestamp of our map ->
  // odom transform correctly
  last_scan_timestamp.reset();
  // this is the current scan that our run_loop should process
  scan_to_process.reset();

  timer = this->create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&ParticleFilter::pub_latest_transform, this));

  std::random_device rd; // get random initial number (seed)
  random_generator_.seed(rd()); // use seed with random number generator
}

void ParticleFilter::pub_latest_transform()
{
  if (last_scan_timestamp.has_value())
  {
    rclcpp::Time last_scan_time(last_scan_timestamp.value());
    rclcpp::Duration offset(0, 100000000);
    auto postdated_timestamp = last_scan_time + offset;
    transform_helper_->send_last_map_to_odom_transform(map_frame, odom_frame,
                                                       postdated_timestamp);
  }
}

void ParticleFilter::run_loop()
{
  if (!scan_to_process.has_value())
  {
    return;
  }
  auto msg = scan_to_process.value();
  std::tuple<std::optional<geometry_msgs::msg::Pose>, std::optional<std::chrono::nanoseconds>>
      matching_odom_pose = transform_helper_->get_matching_odom_pose(
          odom_frame, base_frame, msg.header.stamp);
  auto new_pose = std::get<0>(matching_odom_pose);
  auto dt = std::get<1>(matching_odom_pose);
  if (!new_pose.has_value())
  {
    // we were unable to get the pose of the robot corresponding to the
    // scan timestamp
    if (dt.has_value() && dt.value() < std::chrono::nanoseconds(0))
    {
      //  we will never get this transform, since it is before our
      //  oldest one
      scan_to_process.reset();
    }
    return;
  }
  auto polar_coord = transform_helper_->convert_scan_to_polar_in_robot_frame(
      msg, base_frame);
  auto r = std::get<0>(polar_coord);
  auto theta = std::get<1>(polar_coord);
  // clear the current scan so that we can process the next one
  scan_to_process.reset();
  odom_pose = new_pose;
  auto new_odom_xy_theta =
      transform_helper_->convert_pose_to_xy_theta(odom_pose.value());
  if (current_odom_xy_theta.size() == 0)
  {
    current_odom_xy_theta = new_odom_xy_theta;
  }
  else if (particle_cloud.size() == 0)
  {
    // now that we have all of the necessary transforms we can update
    // the particle cloud
    initialize_particle_cloud();
  }
  else if (moved_far_enough_to_update(new_odom_xy_theta))
  {
    // we have moved far enough to do an update!
    update_particles_with_odom(); // update based on odometry
    update_particles_with_laser(r,theta); // update based on laser scan
    update_robot_pose();                // update robot's pose based on particles
    resample_particles();               // resample particles to focus on areas of
                                        // high density
  }

  // publish particles (so things like rviz can see them)
  publish_particles(msg.header.stamp);
}

bool ParticleFilter::moved_far_enough_to_update(std::vector<float> new_odom_xy_theta)
{
  return abs(new_odom_xy_theta[0] - current_odom_xy_theta[0] > d_thresh ||
             abs(new_odom_xy_theta[1] - current_odom_xy_theta[1]) >
                 d_thresh ||
             abs(new_odom_xy_theta[2] - current_odom_xy_theta[2]) > a_thresh);
}

void ParticleFilter::update_robot_pose()
{
  // first make sure that the particle weights are normalized
  normalize_particles();


  // assigns the latest pose estimate into self.robot_pose as a geometry_msgs.Pose object
  geometry_msgs::msg::Pose robot_pose;

  
  double total_x = 0.0;
  double total_y = 0.0;
  double avg_cos = 0.0;
  double avg_sin = 0.0;

  for (const auto& p : particle_cloud) {
      // Sum for position
      total_x += p.w * p.x;
      total_y += p.w * p.y;
      
      // Sum for angle
      avg_cos += p.w * std::cos(p.theta);
      avg_sin += p.w * std::sin(p.theta);
  }

  // assign position values to robot pose
  robot_pose.position.x = total_x;
  robot_pose.position.y = total_y;

  // assign thate value to robot pose
  float mean_theta = std::atan2(avg_sin, avg_cos);
  robot_pose.orientation = quaternion_from_euler(0, 0, mean_theta);
  
  if (odom_pose.has_value()) // then update robot pose
  {
    transform_helper_->fix_map_to_odom_transform(robot_pose,odom_pose.value());
  }
  else
  {
    std::cout<< "Pose in the odometry frame has not been set" <<std::endl;
  }
}

void ParticleFilter::update_particles_with_odom()
{
  auto new_odom_xy_theta = transform_helper_->convert_pose_to_xy_theta(odom_pose.value());

  // compute the change in x,y,theta since our last update
  if (current_odom_xy_theta.size() >= 3) /// why >=3 vs ==3
  {
    auto old_odom_xy_theta = current_odom_xy_theta;
    auto delta_x = new_odom_xy_theta[0] - current_odom_xy_theta[0];
    auto delta_y = new_odom_xy_theta[1] - current_odom_xy_theta[1];
    auto delta_theta = new_odom_xy_theta[2] - current_odom_xy_theta[2];

    // for each particle in particles, change in x by delta_x, y by delta_y, theta by delta_theta
    // for (Particle& p : particle_cloud) {
    //   p.x += delta_x;
    //   p.y += delta_y;
    //   p.theta += delta_theta;
    // }

    // In update_particles_with_odom()
    for (Particle& p : particle_cloud) {
      // Correct motion model:
      // Rotate the odom-frame delta by the particle's map-frame heading
      float cos_theta = std::cos(p.theta);
      float sin_theta = std::sin(p.theta);
      
      p.x += delta_x * cos_theta - delta_y * sin_theta;
      p.y += delta_x * sin_theta + delta_y * cos_theta;
      p.theta += delta_theta;
      p.theta = angles::normalize_angle(p.theta); // Always normalize angles!
    }

    // surely this works
    // For some reason I think the starter code didn't reset the odom distance
    // after deciding the odom distance was far enough to update the particles
    // so I'm doing that here
     current_odom_xy_theta = new_odom_xy_theta;
 
  }
  else
  {
    current_odom_xy_theta = new_odom_xy_theta;
    return;
  }

  // TODO: test this
  check_particles_inbounds();
}

void ParticleFilter::resample_particles()
{
  // make sure the distribution is normalized
  normalize_particles();

  // Define percentages
  const double truncation_percentage = truncation_percentage_; // Remove bottom x%
  const double random_percentage = random_percentage_;      // Add x% new random particles

  // Calculate particle group counts
  size_t num_to_truncate = static_cast<size_t>(this->n_particles * truncation_percentage);
  size_t num_random = static_cast<size_t>(this->n_particles * random_percentage);
  size_t num_duplicates_to_add = num_to_truncate - num_random;
  size_t num_to_remove = static_cast<size_t>(this->n_particles * truncation_percentage);

  // Sort particles by weight in ascending order (lowest weight first)
  std::sort(particle_cloud.begin(), particle_cloud.end(),
            [](const Particle& a, const Particle& b) {
              return a.w < b.w;
            });

  // Remove the lowest-weighted particles from the vector
  particle_cloud.erase(particle_cloud.begin(), particle_cloud.begin() + num_to_remove);


  // Duplicate the highest-weighted particles
  std::vector<Particle> new_particles;
  new_particles.reserve(num_to_remove);

  // Re-calculate the sum of weights for the surviving particles (the top 80%)
  float survivor_weight_sum = 0.0f;
  for (const auto& p : particle_cloud) {
    survivor_weight_sum += p.w;
  }

  // Create noise generators
  std::normal_distribution<float> x_noise(0.0, static_cast<float>(resample_noise_x_stddev_));
  std::normal_distribution<float> y_noise(0.0, static_cast<float>(resample_noise_y_stddev_));
  std::normal_distribution<float> theta_noise(0.0, static_cast<float>(resample_noise_theta_stddev_));

  // Determine how many duplicates each survivor should generate (20% of all particles will be dispersed)
  for (const auto& survivor : particle_cloud) {
    // The number of new particles this survivor will spawn is proportional to its weight
    // relative to the other survivors.
    long num_duplicates = std::round((survivor.w / survivor_weight_sum) * (num_to_remove*0.8));

    for (long i = 0; i < num_duplicates; ++i) {
      Particle new_particle = survivor; // Create a copy

      // Add noise to the duplicated particles
      new_particle.x += x_noise(random_generator_);
      new_particle.y += y_noise(random_generator_);
      new_particle.theta += theta_noise(random_generator_);
      new_particle.theta = angles::normalize_angle(new_particle.theta);

      new_particles.push_back(new_particle);
      
    }
  }

  // remaing particles are all random
  while (new_particles.size() < num_to_truncate) {
    new_particles.push_back(random_particle());
  }

  particle_cloud.insert(particle_cloud.end(), new_particles.begin(), new_particles.end());
  check_particles_inbounds();
}

void ParticleFilter::update_particles_with_laser(std::vector<float> r, std::vector<float> theta)
{
  /*
  - get laser scan data
  - determine laser scan closest distance (cd_l) to real robot, that is above a threshold (and store the corresponding angle as theta_l)

  - project this closest distance (robot_cd_l) at corresponding angle (theta_l) from each particle
  - find closest distance from projection to obstacle (using helper)
  - the closer that returned distance is to 0, the higher the weight
  
  - check if inbounds
  - call normalize
  */

  // Determine laser scan closest distance (cd_l) and angle (theta_l)
  float cd_l = std::numeric_limits<float>::infinity();
  float theta_l = 0.0f;
  const float distance_threshold = 0.1f; // minimum distance threshold in meters
  
  for (size_t i = 0; i < r.size(); i++) {
    // Only consider finite readings above threshold
    if (std::isfinite(r[i]) && r[i] > distance_threshold) {
      if (r[i] < cd_l) {
        cd_l = r[i];
        theta_l = theta[i];
      }
    }
  }
  
  // If no valid closest distance found, skip update
  if (!std::isfinite(cd_l)) {
    return;
  }
  
  // For each particle, determine weight
  for (Particle& particle : particle_cloud) {
    
    // Convert particle coords to polar
    // Project cd_l at angle theta_l from the particle's position and orientation
    float ang = particle.theta + theta_l;
    float endpoint_x = particle.x + cd_l * std::cos(ang);
    float endpoint_y = particle.y + cd_l * std::sin(ang);
    
    // Get distance to closest obstacle on the map from this projected point (cd_proj)
    double cd_proj = occupancy_field->get_closest_obstacle_distance(endpoint_x, endpoint_y);
    
    // Calculate weight based on difference between laser and particle measurements
    if (std::isfinite(cd_proj)) {
      double sigma = 1.0;
      particle.w = std::exp(-(cd_proj * cd_proj) / (2.0 * sigma * sigma));
    } else {
      // If measurement is invalid, assign epsilon to avoid x/0
      particle.w = 0.0001f;
    }
  }
  
  // Affirm particles are in valid places and normalize weights
  check_particles_inbounds();
  normalize_particles();

  (void)r;
  (void)theta;
}

void ParticleFilter::update_initial_pose(geometry_msgs::msg::PoseWithCovarianceStamped msg)
{
  auto xy_theta = transform_helper_->convert_pose_to_xy_theta(msg.pose.pose);
  initialize_particle_cloud(xy_theta);
}

void ParticleFilter::initialize_particle_cloud(
    std::optional<std::vector<float>> xy_theta)
{
  // reset cloud
  particle_cloud.clear();
  particle_cloud.reserve(this->n_particles);

  // Define standard deviations for initial pose
  const float init_x_stddev = 0.5;
  const float init_y_stddev = 0.5;
  const float init_theta_stddev = M_PI / 12.0;

  // Set up random distributions
  std::normal_distribution<float> x_noise(0.0, init_x_stddev);
  std::normal_distribution<float> y_noise(0.0, init_y_stddev);
  std::normal_distribution<float> theta_noise(0.0, init_theta_stddev);

  // where to initialize the particle cloud
  if (xy_theta.has_value()) {
    RCLCPP_INFO(this->get_logger(), "Initializing particles around provided pose.");
    float mean_x = xy_theta.value()[0];
    float mean_y = xy_theta.value()[1];
    float mean_theta = xy_theta.value()[2];

    for (int i = 0; i < this->n_particles; i++) {
        float x = mean_x + x_noise(random_generator_);
        float y = mean_y + y_noise(random_generator_);
        float theta = mean_theta + theta_noise(random_generator_);
        particle_cloud.push_back(Particle(1.0f / this->n_particles, theta, x, y));
    }
  }
  else {
    RCLCPP_INFO(this->get_logger(), "Initializing particles globally.");
    // This is so you don't need to pass in the odom pose everytime.
    // (Note: this branch is only used for the *very first* localization)
    for (int i = 0; i < this->n_particles; i++) {
        this->particle_cloud.push_back(this->random_particle());
    }
  }

  normalize_particles(); // Maybe remove this since update_robot_pose also does this
  update_robot_pose();
}

Particle ParticleFilter::random_particle() {
  std::array<double, 4> bounds = occupancy_field->get_obstacle_bounding_box();
  float lx = bounds[0];
  float ux = bounds[1];
  float ly = bounds[2];
  float uy = bounds[3];    
  
  float width = ux - lx;
  float height = uy - ly;

  float x, y, theta;
  float w = 1.0f / this->n_particles;

  while (true) {
    // use uniform distribution across entire map
    float random_val_1 = uniform_distribution_(random_generator_);
    float random_val_2 = uniform_distribution_(random_generator_);
    float random_val_3 = uniform_distribution_(random_generator_);
    
    x = lx + width * random_val_1;
    y = ly + height * random_val_2;
    theta = 2.0f * M_PI * random_val_3;

    // get closest distance to obstacle and check if valid distance (not infinite, not inside obstacle)
    float dist = occupancy_field->get_closest_obstacle_distance(x, y);

    if (std::isfinite(dist) && dist > 0.0) {
      break;
    }
  }

  return Particle(w, theta, x, y);
}

void ParticleFilter::check_particles_inbounds() {
  std::array<double, 4> bounds = occupancy_field->get_obstacle_bounding_box();
  float lx = bounds[0];
  float ux = bounds[1];
  float ly = bounds[2];
  float uy = bounds[3];    

  if (lx > ux) std::swap(lx, ux);
  if (ly > uy) std::swap(ly, uy);

  for (Particle& p : particle_cloud) {
    // check if particle i is in bounds
//    if (lx <= p.x || p.x <= ux || ly <= p.y || p.y <= uy) {
    if (p.x < lx || p.x > ux || p.y < ly || p.y > uy) {
      p = random_particle();
      // test
      // p = Particle(1.0f/this->n_particles, 0.0f, 0.0f, 0.0f);
    }
  }
}

void ParticleFilter::normalize_particles()
{
  // TODO: test this
  // Sum of all weights divided by number of all particles
  // for particle in particles, divide by average of weights
  float sum_weights = 0;
  for (int i = 0; i < n_particles; i ++) {
    sum_weights += particle_cloud[i].w;
  }
    
  for (int i = 0; i < n_particles; i ++) {
    particle_cloud[i].w /= sum_weights; // changed from avg weight to total weight -> I'm 98% sure this is right
  }
  
}

void ParticleFilter::publish_particles(rclcpp::Time timestamp)
{
  nav2_msgs::msg::ParticleCloud msg;
  msg.header.stamp = timestamp;
  msg.header.frame_id = map_frame;

  for (unsigned int i = 0; i < particle_cloud.size(); i++)
  {
    nav2_msgs::msg::Particle converted;
    converted.weight = particle_cloud[i].w;
    converted.pose = particle_cloud[i].as_pose();
    msg.particles.push_back(converted);
//    msg.particle_cloud.push_back(converted); //given code, caused an error
  }

  // actually send the message so that we can view it in rviz
  particle_pub->publish(msg);
}

// Callback function for parameter updates.
rcl_interfaces::msg::SetParametersResult ParticleFilter::on_parameters_changed(
    const std::vector<rclcpp::Parameter> & parameters)
{
  auto result = rcl_interfaces::msg::SetParametersResult();
  result.successful = true;

  // Loop through all parameters that were changed
  for (const auto & param : parameters) {
    std::string name = param.get_name();
    RCLCPP_INFO(get_logger(), "Parameter changed: %s", name.c_str());

    // Check which parameter it is and update the corresponding member variable
    if (name == "n_particles" && param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
      n_particles = param.as_int();
    } else if (name == "d_thresh" && param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
      d_thresh = param.as_double();
    } else if (name == "a_thresh" && param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
      a_thresh = param.as_double();
    } else if (name == "resampling.truncation_percentage" && param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
      truncation_percentage_ = param.as_double();
    } else if (name == "resampling.random_percentage" && param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
      random_percentage_ = param.as_double();
    } else if (name == "resampling.noise_x_stddev" && param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
      resample_noise_x_stddev_ = param.as_double();
    } else if (name == "resampling.noise_y_stddev" && param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
      resample_noise_y_stddev_ = param.as_double();
    } else if (name == "resampling.noise_theta_stddev" && param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
      resample_noise_theta_stddev_ = param.as_double();
    }
  }
  return result;
}

void ParticleFilter::scan_received(sensor_msgs::msg::LaserScan msg)
{
  last_scan_timestamp = msg.header.stamp;
  /**
   * we throw away scans until we are done processing the previous scan
   * self.scan_to_process is set to None in the run_loop
   */
  if (!scan_to_process.has_value())
  {
    scan_to_process = msg;
  }
  // call run_loop to see if we need to update our filter, this will prevent more scans from coming in
  run_loop();
}

void ParticleFilter::setup_helpers(std::shared_ptr<ParticleFilter> nodePtr)
{
  occupancy_field = std::make_shared<OccupancyField>(OccupancyField(nodePtr));
  std::cout << "done generating occupancy field" << std::endl;
  transform_helper_ = std::make_shared<TFHelper>(TFHelper(nodePtr));
  std::cout << "done generating TFHelper" << std::endl;
}

int main(int argc, char **argv)
{
  // this is useful to give time for the map server to get ready...
  // TODO: fix in some other way
  sleep(5);
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<ParticleFilter>();
  node->setup_helpers(node);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
