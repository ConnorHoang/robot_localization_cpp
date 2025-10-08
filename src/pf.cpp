#define _USE_MATH_DEFINES

#include <cmath>
#include <iostream>
#include <rclcpp/time.hpp>
#include <string>
#include <tuple>
#include <random>

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

  n_particles = 300; // the number of particles to use

  d_thresh = 0.2; // the amount of linear movement before performing an update
  a_thresh =
      M_PI / 6; // the amount of angular movement before performing an update

  // TODO: define additional constants if needed

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

  std::random_device rd; // get random seed
  random_generator_.seed(rd()); // use seed
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

  //TODO: Test this

  // first make sure that the particle weights are normalized
  normalize_particles();

  // determine best current pose estimate as lowest weight particle(closest to real data)
  int index_of_lowest_weight = 0;
  float lowest_weight = particle_cloud[0].w;
  for (int i = 1; i < n_particles; i ++) {
    if (particle_cloud[i].w < lowest_weight) {
      index_of_lowest_weight = 0;
      lowest_weight = particle_cloud[i].w;
    }
  }  

  // assigns the latest pose estimate into self.robot_pose as a geometry_msgs.Pose object
  geometry_msgs::msg::Pose robot_pose;
  robot_pose.position.x = particle_cloud[index_of_lowest_weight].x;
  robot_pose.position.y = particle_cloud[index_of_lowest_weight].y;
  // might be wrong
  robot_pose.orientation = quaternion_from_euler(particle_cloud[index_of_lowest_weight].theta, 0.0, 0.0);
  
  if (odom_pose.has_value()) // then update robot pose
  {
    transform_helper_->fix_map_to_odom_transform(robot_pose,
                                                 odom_pose.value());
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
  if (current_odom_xy_theta.size() >= 3)
  {
    auto old_odom_xy_theta = current_odom_xy_theta;
    auto delta_x = new_odom_xy_theta[0] - current_odom_xy_theta[0];
    auto delta_y = new_odom_xy_theta[1] - current_odom_xy_theta[1];
    auto delta_theta = new_odom_xy_theta[2] - current_odom_xy_theta[2];

    // for each particle in particles, change in x by delta_x, y by delta_y, theta by delta_theta
    for (int i = 0; i < n_particles; i ++) {
      particle_cloud[i].x += delta_x;
      particle_cloud[i].y += delta_y;
      particle_cloud[i].theta += delta_theta;
    }  
  }
  else
  {
    current_odom_xy_theta = new_odom_xy_theta;
    return;
  }

  // TODO: test this
}

void ParticleFilter::resample_particles()
{
  // make sure the distribution is normalized
  normalize_particles();
  // TODO: fill out the rest of the implementation
  // select lowest ceiling(n_particles/20) particles based upon weight (remove all others). 
  // Use same method as in particle creation to create weighted distribution of new particles around by how many you remove 
  // (take position of paticle and transform with difference generated using random distribution) (add noise)

  // Remove bottom 20% of existing particles
  const double truncation_percentage = 0.20;
  size_t num_to_remove = static_cast<size_t>(this->n_particles * truncation_percentage);
  size_t num_to_keep = this->n_particles - num_to_remove;

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
  std::normal_distribution<float> x_noise(0.0, resample_noise_x_stddev_);
  std::normal_distribution<float> y_noise(0.0, resample_noise_y_stddev_);
  std::normal_distribution<float> theta_noise(0.0, resample_noise_theta_stddev_);

  // Determine how many duplicates each survivor should generate
  for (const auto& survivor : particle_cloud) {
    // The number of new particles this survivor will spawn is proportional to its weight
    // relative to the other survivors.
    long num_duplicates = std::round((survivor.w / survivor_weight_sum) * num_to_remove);

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

  // Deal with missing particle
  while (new_particles.size() < num_to_remove) {
    // If we are short, duplicate the best particle (which is now at the end of the sorted vector)
    Particle best_particle = particle_cloud.back();
    best_particle.x += x_noise(random_generator_);
    best_particle.y += y_noise(random_generator_);
    best_particle.theta += theta_noise(random_generator_);
    best_particle.theta = angles::normalize_angle(best_particle.theta);
    new_particles.push_back(best_particle);
  }
  // Deal with extra particle
  while (new_particles.size() > num_to_remove) {
    new_particles.pop_back(); // If we have too many, remove the last-added ones
  }

  particle_cloud.insert(particle_cloud.end(), new_particles.begin(), new_particles.end());
}

void ParticleFilter::update_particles_with_laser(std::vector<float> r, std::vector<float> theta)
{
  // TODO: implement this

  /*
  get laser scan data
  determine laser scan closest distance (cd_l) to object, that is above a threshold (ros can give this)
  determine laser scan angle to closest object (theta_l)
  determine each particle closest distance (cd_p)
  determine each particle angle to closest distance (theta_p) <- actually just use the given one
  weight = abs(sqrt((cd_l-cd_p)^2+(theta_l-theta_p)^2))
  call normalize
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
  
  // For each particle, determine closest distance (cd_p)
  for (size_t p = 0; p < particle_cloud.size(); p++) {
    Particle& particle = particle_cloud[p];
    
    // Use the same angle as the laser scan (theta_l)
    float theta_p = theta_l;
    
    // Calculate the endpoint position in map frame for this particle
    float ang = particle.theta + theta_p;
    float endpoint_x = particle.x + cd_l * std::cos(ang);
    float endpoint_y = particle.y + cd_l * std::sin(ang);
    
    // Get distance to closest obstacle from this endpoint (cd_p)
    double cd_p = occupancy_field->get_closest_obstacle_distance(
        endpoint_x, endpoint_y);
    
    // Calculate weight based on difference between laser and particle measurements
    if (std::isfinite(cd_p)) {
      // weight = 1 / abs(sqrt((cd_l - cd_p)^2 + (theta_l - theta_p)^2))
      // theta_p = theta_l, the angle difference is 0
      float distance_diff = cd_l - static_cast<float>(cd_p);
      float deviation = std::abs(distance_diff);
      
      // Weight is inversely proportional to deviation
      // Add small epsilon (yay discrete) to avoid division by zero
      particle.w = 1.0f / (deviation + 0.001f);
    } else {
      // If no valid measurement, assign epsilon weight
      particle.w = 0.0001f;
    }
  }
  
  // Normalize particle weights
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
    std::optional<std::vector<float>> xy_theta = )
{
  // where to initialize the particle cloud
  if (!xy_theta.has_value())
  {
    // This is so you don't need to pass in the odom pose everytime.
    xy_theta = transform_helper_->convert_pose_to_xy_theta(odom_pose.value());
  }

  // TODO: create normal distribution of particles around this point
  particle_cloud.clear();
  particle_cloud.reserve(this->n_particles);

  for (size_t i = 0; i < this->n_particles; i++) {
    this->particle_cloud.push_back(this->random_particle());
  }

  normalize_particles(); // Maybe remove this since update_robot_pose also does this
  update_robot_pose();
}

auto ParticleFilter::random_particle() {
  // return random particle
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
    float random_val_1 = uniform_distribution_(random_generator_);
    float random_val_2 = uniform_distribution_(random_generator_);
    float random_val_3 = uniform_distribution_(random_generator_);
    
    x = lx + width * random_val_1;
    y = ly + height * random_val_2;
    theta = 2.0f * M_PI * random_val_3;

    if (std::isfinite(occupancy_field->get_closest_obstacle_distance(x, y))) {
      break;
    }
  }

  return Particle(w, theta, x, y);
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
  
  float avg_weight = sum_weights / n_particles;
  
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
    msg.particle_cloud.push_back(converted);
  }

  // actually send the message so that we can view it in rviz
  particle_pub->publish(msg);
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