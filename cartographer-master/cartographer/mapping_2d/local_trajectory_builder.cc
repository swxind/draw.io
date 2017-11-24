/*
 * Copyright 2016 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cartographer/mapping_2d/local_trajectory_builder.h"

#include <limits>

#include "cartographer/common/make_unique.h"
#include "cartographer/sensor/range_data.h"

namespace cartographer {
namespace mapping_2d {

LocalTrajectoryBuilder::LocalTrajectoryBuilder(
    const proto::LocalTrajectoryBuilderOptions& options)
    : options_(options),
      active_submaps_(options.submaps_options()),
      motion_filter_(options_.motion_filter_options()),
      real_time_correlative_scan_matcher_(
          options_.real_time_correlative_scan_matcher_options()),
      ceres_scan_matcher_(options_.ceres_scan_matcher_options()),
      odometry_state_tracker_(options_.num_odometry_states()) {}

LocalTrajectoryBuilder::~LocalTrajectoryBuilder() {}

sensor::RangeData LocalTrajectoryBuilder::TransformAndFilterRangeData(
    const transform::Rigid3f& tracking_to_tracking_2d,
    const sensor::RangeData& range_data) const {
  const sensor::RangeData cropped = sensor::CropRangeData(
      sensor::TransformRangeData(range_data, tracking_to_tracking_2d),
      options_.min_z(), options_.max_z());
  return sensor::RangeData{
      cropped.origin,
      sensor::VoxelFiltered(cropped.returns, options_.voxel_filter_size()),
      sensor::VoxelFiltered(cropped.misses, options_.voxel_filter_size())};
}

/*
首先对激光数据进行了投射处理、栅格化滤波，
然后实时在线优化real_time_correlative_scan_matcher_.Match，这个优化是可选的，优化方式比较简单，
   (就是在之前预测位姿附近开一个三维窗口，
	然后在这个三维窗口内，均匀播撒候选粒子，对每个粒子计算匹配得分，选取最高得分的粒子作为优化的位姿)
该优化的位姿还要传入一个ceres_scan_matcher_.Match(）在子地图中完成局部优化，最终得到本地优化后的位姿。
*/
void LocalTrajectoryBuilder::ScanMatch(
    common::Time time, const transform::Rigid3d& pose_prediction,
    const transform::Rigid3d& tracking_to_tracking_2d,
    const sensor::RangeData& range_data_in_tracking_2d,
    transform::Rigid3d* pose_observation) {
    
  std::shared_ptr<const Submap> matching_submap =
      active_submaps_.submaps().front();

  //根据参数，得到一个pose预测 pose_prediction来自odom，tracking_to_tracking_2d角度预测
  transform::Rigid2d pose_prediction_2d =
      transform::Project2D(pose_prediction * tracking_to_tracking_2d.inverse());
  
  // The online correlative scan matcher will refine the initial estimate for
  // the Ceres scan matcher.
  transform::Rigid2d initial_ceres_pose = pose_prediction_2d;
  
  sensor::AdaptiveVoxelFilter adaptive_voxel_filter(
      options_.adaptive_voxel_filter_options());  // 根据参数，对点云数据进行 VoxelFilter(对点云数据进行处理)
  
  const sensor::PointCloud filtered_point_cloud_in_tracking_2d =
      adaptive_voxel_filter.Filter(range_data_in_tracking_2d.returns); //获取处理后的 激光测距 的点云数据；
  
  if (options_.use_online_correlative_scan_matching()) {			//根据参数判断是否调用该方法
    real_time_correlative_scan_matcher_.Match(                                   //调用scan_matching中的匹配方法，得到最优的pose_estimate；
        pose_prediction_2d, filtered_point_cloud_in_tracking_2d,
        matching_submap->probability_grid(), &initial_ceres_pose);
  }

  transform::Rigid2d tracking_2d_to_map;
  ceres::Solver::Summary summary;
  ceres_scan_matcher_.Match(pose_prediction_2d, initial_ceres_pose,       //本地优化过程
                            filtered_point_cloud_in_tracking_2d,
                            matching_submap->probability_grid(),
                            &tracking_2d_to_map, &summary);

  *pose_observation =
      transform::Embed3D(tracking_2d_to_map) * tracking_to_tracking_2d;   //scan_match 返回值，本地优化后的位姿
}


/*
GlobalTrajectoryBuilder::AddRangefinderData的本地优化；
会调用AddAccumulatedRangeData，其中调用ScanMatch；
*/
std::unique_ptr<LocalTrajectoryBuilder::InsertionResult>
LocalTrajectoryBuilder::AddHorizontalRangeData(
    const common::Time time, const sensor::RangeData& range_data) {
  // Initialize IMU tracker now if we do not ever use an IMU.
  if (!options_.use_imu_data()) {
    InitializeImuTracker(time);
  }

  if (imu_tracker_ == nullptr) {
    // Until we've initialized the IMU tracker with our first IMU message, we
    // cannot compute the orientation of the rangefinder.
    LOG(INFO) << "ImuTracker not yet initialized.";
    return nullptr;
  }

  Predict(time);  //得到一个pose_estimate_；
  if (num_accumulated_ == 0) {
    first_pose_estimate_ = pose_estimate_.cast<float>();
    accumulated_range_data_ =
        sensor::RangeData{Eigen::Vector3f::Zero(), {}, {}};
  }

  const transform::Rigid3f tracking_delta =
      first_pose_estimate_.inverse() * pose_estimate_.cast<float>();  //得到一个预测的位姿
  
  const sensor::RangeData range_data_in_first_tracking =
      sensor::TransformRangeData(range_data, tracking_delta);    //传感器 + 预测 + 转换 = 在预测的位置下，传感器消息表达的信息
      
  // Drop any returns below the minimum range and convert returns beyond the
  // maximum range into misses.
  for (const Eigen::Vector3f& hit : range_data_in_first_tracking.returns) {   //遍历传感器的消息
    const Eigen::Vector3f delta = hit - range_data_in_first_tracking.origin;	//计算hit 与 激光原点的距离；
    const float range = delta.norm();
    if (range >= options_.min_range()) {						// 根据设定参数做判断 >= min_range <= max_range = accumulated_range_data_.hit
      if (range <= options_.max_range()) {
        accumulated_range_data_.returns.push_back(hit);
      } else {											//  >= min_range >= max_range = accumulated_range_data_.misses
        accumulated_range_data_.misses.push_back(				
            range_data_in_first_tracking.origin +							
            options_.missing_data_ray_length() / range * delta);           // 如果距离大于 >= max_range, 只确保missing_data_ray_length的距离为miss；
      }
    }
  }
  ++num_accumulated_;

  if (num_accumulated_ >= options_.scans_per_accumulation()) {    // 每累计scans_per_accumulation个数据后，进行下一步；
    num_accumulated_ = 0;
    return AddAccumulatedRangeData(
        time, sensor::TransformRangeData(accumulated_range_data_,
                                         tracking_delta.inverse()));
  }
  return nullptr;
}

std::unique_ptr<LocalTrajectoryBuilder::InsertionResult>
LocalTrajectoryBuilder::AddAccumulatedRangeData(
    const common::Time time, const sensor::RangeData& range_data) {

//通过里程计校正量得到一个靠谱一点的里程计预测位置；
  const transform::Rigid3d odometry_prediction =
      pose_estimate_ * odometry_correction_;  
  
  const transform::Rigid3d model_prediction = pose_estimate_;
  // TODO(whess): Prefer IMU over odom orientation if configured?
  const transform::Rigid3d& pose_prediction = odometry_prediction;

  // Computes the rotation without yaw, as defined by GetYaw(). 得到一个旋转角度
  const transform::Rigid3d tracking_to_tracking_2d =
      transform::Rigid3d::Rotation(
          Eigen::Quaterniond(Eigen::AngleAxisd(
              -transform::GetYaw(pose_prediction), Eigen::Vector3d::UnitZ())) *
          pose_prediction.rotation());

  const sensor::RangeData range_data_in_tracking_2d =    
      TransformAndFilterRangeData(tracking_to_tracking_2d.cast<float>(),      // 根据设定的参数，对数据做一定的裁剪
                                  range_data);

  if (range_data_in_tracking_2d.returns.empty()) {
    LOG(WARNING) << "Dropped empty horizontal range data.";
    return nullptr;
  }

  ScanMatch(time, pose_prediction, tracking_to_tracking_2d,                   //调用ScanMatch;  得到一个较为精准的pose_estimate_
            range_data_in_tracking_2d, &pose_estimate_);
  
  odometry_correction_ = transform::Rigid3d::Identity();			//用优化后的位姿，校正odom状态
  if (!odometry_state_tracker_.empty()) {
    // We add an odometry state, so that the correction from the scan matching
    // is not removed by the next odometry data we get.
    odometry_state_tracker_.AddOdometryState(
        {time, odometry_state_tracker_.newest().odometer_pose,
         odometry_state_tracker_.newest().state_pose *
             odometry_prediction.inverse() * pose_estimate_});
  }

  // Improve the velocity estimate.
  if (last_scan_match_time_ > common::Time::min() &&
      time > last_scan_match_time_) {
    const double delta_t = common::ToSeconds(time - last_scan_match_time_);
    // This adds the observed difference in velocity that would have reduced the
    // error to zero.
    velocity_estimate_ += (pose_estimate_.translation().head<2>() -       
                           model_prediction.translation().head<2>()) /
                          delta_t;
  }
  last_scan_match_time_ = time_;

  // Remove the untracked z-component which floats around 0 in the UKF.
  const auto translation = pose_estimate_.translation();
  pose_estimate_ = transform::Rigid3d(
      transform::Rigid3d::Vector(translation.x(), translation.y(), 0.),
      pose_estimate_.rotation());

  const transform::Rigid3d tracking_2d_to_map =
      pose_estimate_ * tracking_to_tracking_2d.inverse();
  
  last_pose_estimate_ = {
      time, pose_estimate_,
      sensor::TransformPointCloud(range_data_in_tracking_2d.returns,
                                  tracking_2d_to_map.cast<float>())};

  const transform::Rigid2d pose_estimate_2d =
      transform::Project2D(tracking_2d_to_map);
  
  if (motion_filter_.IsSimilar(time, transform::Embed3D(pose_estimate_2d))) {  //根据设定参数判断，机器人是否有走动；(时间，位移, 角度)
    return nullptr;
  }

  // Querying the active submaps must be done here before calling
  // InsertRangeData() since the queried values are valid for next insertion.
  std::vector<std::shared_ptr<const Submap>> insertion_submaps;
  for (std::shared_ptr<Submap> submap : active_submaps_.submaps()) {
    insertion_submaps.push_back(submap);
  }
  active_submaps_.InsertRangeData(
      TransformRangeData(range_data_in_tracking_2d,
                         transform::Embed3D(pose_estimate_2d.cast<float>())));

  return common::make_unique<InsertionResult>(InsertionResult{
      time, std::move(insertion_submaps), tracking_to_tracking_2d,
      range_data_in_tracking_2d, pose_estimate_2d});
}

const LocalTrajectoryBuilder::PoseEstimate&
LocalTrajectoryBuilder::pose_estimate() const {
  return last_pose_estimate_;
}

/*
Predict()函数，这个函数主要是运用运动学模型（陀螺仪的积分，里程计的积分），简单的预测姿态平移矩阵以及旋转矩阵。
预测完了后，又加入了加速度与角速度的观测。
*/
void LocalTrajectoryBuilder::AddImuData(
    const common::Time time, const Eigen::Vector3d& linear_acceleration,
    const Eigen::Vector3d& angular_velocity) {
  CHECK(options_.use_imu_data()) << "An unexpected IMU packet was added.";

  InitializeImuTracker(time);
  Predict(time);
  imu_tracker_->AddImuLinearAccelerationObservation(linear_acceleration);
  imu_tracker_->AddImuAngularVelocityObservation(angular_velocity);
}


/*
首先执行了predict（），然后使用里程计的数据，得到了一个里程计的校正量odometry_correction_。
*/
void LocalTrajectoryBuilder::AddOdometerData(
    const common::Time time, const transform::Rigid3d& odometer_pose) {
  if (imu_tracker_ == nullptr) {
    // Until we've initialized the IMU tracker we do not want to call Predict().
    LOG(INFO) << "ImuTracker not yet initialized.";
    return;
  }

  Predict(time);
  if (!odometry_state_tracker_.empty()) {
    const auto& previous_odometry_state = odometry_state_tracker_.newest();
    const transform::Rigid3d delta =
        previous_odometry_state.odometer_pose.inverse() * odometer_pose;
    const transform::Rigid3d new_pose =
        previous_odometry_state.state_pose * delta;
    odometry_correction_ = pose_estimate_.inverse() * new_pose;
  }
  odometry_state_tracker_.AddOdometryState(
      {time, odometer_pose, pose_estimate_ * odometry_correction_});
}

void LocalTrajectoryBuilder::InitializeImuTracker(const common::Time time) {
  if (imu_tracker_ == nullptr) {
    imu_tracker_ = common::make_unique<mapping::ImuTracker>(
        options_.imu_gravity_time_constant(), time);
  }
}

void LocalTrajectoryBuilder::Predict(const common::Time time) {
  CHECK(imu_tracker_ != nullptr);
  CHECK_LE(time_, time);

  //从IMU中获取 方向；
  const double last_yaw = transform::GetYaw(imu_tracker_->orientation());

  imu_tracker_->Advance(time);
  if (time_ > common::Time::min()) {
    const double delta_t = common::ToSeconds(time - time_);

// Constant velocity model.当做一个 恒速运动模型；
// 位移= x0 + v*t;
    const Eigen::Vector3d translation =
        pose_estimate_.translation() +
        delta_t *
            Eigen::Vector3d(velocity_estimate_.x(), velocity_estimate_.y(), 0.);

// Use the current IMU tracker roll and pitch for gravity alignment, and
// apply its change in yaw.
// 旋转 = 
    const Eigen::Quaterniond rotation =
        Eigen::AngleAxisd(
            transform::GetYaw(pose_estimate_.rotation()) - last_yaw,
            Eigen::Vector3d::UnitZ()) *
        imu_tracker_->orientation();
    pose_estimate_ = transform::Rigid3d(translation, rotation);
  }
  time_ = time;
}

}  // namespace mapping_2d
}  // namespace cartographer
