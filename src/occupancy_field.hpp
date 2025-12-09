#ifndef OCCUPANCYFIELD_HPP
#define OCCUPANCYFIELD_HPP

#include <nav_msgs/msg/occupancy_grid.hpp>

#include "knncpp.h"
#include "rclcpp/rclcpp.hpp"

typedef Eigen::MatrixXd Matrix;

/**
 * Stores an occupancy field for an input map.  An occupancy field returns
 * the distance to the closest obstacle for any coordinate in the map
 * Attributes:
 *      map: the map to localize against (nav_msgs/OccupancyGrid)
 *      closest_occ: the distance for each entry in the OccupancyGrid to
 *      the closest obstacle
 */
class OccupancyField {
public:
  nav_msgs::msg::OccupancyGrid map;
  Matrix closest_occ;
  Matrix occupied_coordinates;

  OccupancyField(std::shared_ptr<rclcpp::Node> node);

  /**
   * Returns: the upper and lower bounds of x and y such that the resultant
   * bounding box contains all of the obstacles in the map.  The format of
   * the return value is ((x_lower, x_upper), (y_lower, y_upper))
   * 
   * If customBounds is true, the bounding box will be limited to the specified custom_x_max=530 and custom_y_max=1433 values which 
   * are approximately the dimensions of Olin college's MAC as kept in the bag files under the maps folder.
   * 
   * @param unit_val_max The maximum occupancy value to consider a cell occupied (default 16)
   * @param customBounds Whether to use custom bounding box limits (default false)
   * @param custom_x_max The custom maximum x value for the bounding box (default 530)
   * @param custom_y_max The custom maximum y value for the bounding box (default 1433)
   */
  std::array<double, 4> get_obstacle_bounding_box(unsigned int unit_val_max=16, bool customBounds=false, 
    unsigned int custom_x_max=530, unsigned int custom_y_max=1433);

  /**
   * Compute the closest obstacle to the specified (x,y) coordinate in
   * the map.  If the (x,y) coordinate is out of the map boundaries, nan
   * will be returned.
   */
  double get_closest_obstacle_distance(float x, float y);
};
#endif
