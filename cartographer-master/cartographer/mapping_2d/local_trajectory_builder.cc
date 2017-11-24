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
���ȶԼ������ݽ�����Ͷ�䴦��դ���˲���
Ȼ��ʵʱ�����Ż�real_time_correlative_scan_matcher_.Match������Ż��ǿ�ѡ�ģ��Ż���ʽ�Ƚϼ򵥣�
   (������֮ǰԤ��λ�˸�����һ����ά���ڣ�
	Ȼ���������ά�����ڣ����Ȳ�����ѡ���ӣ���ÿ�����Ӽ���ƥ��÷֣�ѡȡ��ߵ÷ֵ�������Ϊ�Ż���λ��)
���Ż���λ�˻�Ҫ����һ��ceres_scan_matcher_.Match(�����ӵ�ͼ����ɾֲ��Ż������յõ������Ż����λ�ˡ�
*/
void LocalTrajectoryBuilder::ScanMatch(
    common::Time time, const transform::Rigid3d& pose_prediction,
    const transform::Rigid3d& tracking_to_tracking_2d,
    const sensor::RangeData& range_data_in_tracking_2d,
    transform::Rigid3d* pose_observation) {
    
  std::shared_ptr<const Submap> matching_submap =
      active_submaps_.submaps().front();

  //���ݲ������õ�һ��poseԤ�� pose_prediction����odom��tracking_to_tracking_2d�Ƕ�Ԥ��
  transform::Rigid2d pose_prediction_2d =
      transform::Project2D(pose_prediction * tracking_to_tracking_2d.inverse());
  
  // The online correlative scan matcher will refine the initial estimate for
  // the Ceres scan matcher.
  transform::Rigid2d initial_ceres_pose = pose_prediction_2d;
  
  sensor::AdaptiveVoxelFilter adaptive_voxel_filter(
      options_.adaptive_voxel_filter_options());  // ���ݲ������Ե������ݽ��� VoxelFilter(�Ե������ݽ��д���)
  
  const sensor::PointCloud filtered_point_cloud_in_tracking_2d =
      adaptive_voxel_filter.Filter(range_data_in_tracking_2d.returns); //��ȡ������ ������ �ĵ������ݣ�
  
  if (options_.use_online_correlative_scan_matching()) {			//���ݲ����ж��Ƿ���ø÷���
    real_time_correlative_scan_matcher_.Match(                                   //����scan_matching�е�ƥ�䷽�����õ����ŵ�pose_estimate��
        pose_prediction_2d, filtered_point_cloud_in_tracking_2d,
        matching_submap->probability_grid(), &initial_ceres_pose);
  }

  transform::Rigid2d tracking_2d_to_map;
  ceres::Solver::Summary summary;
  ceres_scan_matcher_.Match(pose_prediction_2d, initial_ceres_pose,       //�����Ż�����
                            filtered_point_cloud_in_tracking_2d,
                            matching_submap->probability_grid(),
                            &tracking_2d_to_map, &summary);

  *pose_observation =
      transform::Embed3D(tracking_2d_to_map) * tracking_to_tracking_2d;   //scan_match ����ֵ�������Ż����λ��
}


/*
GlobalTrajectoryBuilder::AddRangefinderData�ı����Ż���
�����AddAccumulatedRangeData�����е���ScanMatch��
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

  Predict(time);  //�õ�һ��pose_estimate_��
  if (num_accumulated_ == 0) {
    first_pose_estimate_ = pose_estimate_.cast<float>();
    accumulated_range_data_ =
        sensor::RangeData{Eigen::Vector3f::Zero(), {}, {}};
  }

  const transform::Rigid3f tracking_delta =
      first_pose_estimate_.inverse() * pose_estimate_.cast<float>();  //�õ�һ��Ԥ���λ��
  
  const sensor::RangeData range_data_in_first_tracking =
      sensor::TransformRangeData(range_data, tracking_delta);    //������ + Ԥ�� + ת�� = ��Ԥ���λ���£���������Ϣ������Ϣ
      
  // Drop any returns below the minimum range and convert returns beyond the
  // maximum range into misses.
  for (const Eigen::Vector3f& hit : range_data_in_first_tracking.returns) {   //��������������Ϣ
    const Eigen::Vector3f delta = hit - range_data_in_first_tracking.origin;	//����hit �� ����ԭ��ľ��룻
    const float range = delta.norm();
    if (range >= options_.min_range()) {						// �����趨�������ж� >= min_range <= max_range = accumulated_range_data_.hit
      if (range <= options_.max_range()) {
        accumulated_range_data_.returns.push_back(hit);
      } else {											//  >= min_range >= max_range = accumulated_range_data_.misses
        accumulated_range_data_.misses.push_back(				
            range_data_in_first_tracking.origin +							
            options_.missing_data_ray_length() / range * delta);           // ���������� >= max_range, ֻȷ��missing_data_ray_length�ľ���Ϊmiss��
      }
    }
  }
  ++num_accumulated_;

  if (num_accumulated_ >= options_.scans_per_accumulation()) {    // ÿ�ۼ�scans_per_accumulation�����ݺ󣬽�����һ����
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

//ͨ����̼�У�����õ�һ������һ�����̼�Ԥ��λ�ã�
  const transform::Rigid3d odometry_prediction =
      pose_estimate_ * odometry_correction_;  
  
  const transform::Rigid3d model_prediction = pose_estimate_;
  // TODO(whess): Prefer IMU over odom orientation if configured?
  const transform::Rigid3d& pose_prediction = odometry_prediction;

  // Computes the rotation without yaw, as defined by GetYaw(). �õ�һ����ת�Ƕ�
  const transform::Rigid3d tracking_to_tracking_2d =
      transform::Rigid3d::Rotation(
          Eigen::Quaterniond(Eigen::AngleAxisd(
              -transform::GetYaw(pose_prediction), Eigen::Vector3d::UnitZ())) *
          pose_prediction.rotation());

  const sensor::RangeData range_data_in_tracking_2d =    
      TransformAndFilterRangeData(tracking_to_tracking_2d.cast<float>(),      // �����趨�Ĳ�������������һ���Ĳü�
                                  range_data);

  if (range_data_in_tracking_2d.returns.empty()) {
    LOG(WARNING) << "Dropped empty horizontal range data.";
    return nullptr;
  }

  ScanMatch(time, pose_prediction, tracking_to_tracking_2d,                   //����ScanMatch;  �õ�һ����Ϊ��׼��pose_estimate_
            range_data_in_tracking_2d, &pose_estimate_);
  
  odometry_correction_ = transform::Rigid3d::Identity();			//���Ż����λ�ˣ�У��odom״̬
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
  
  if (motion_filter_.IsSimilar(time, transform::Embed3D(pose_estimate_2d))) {  //�����趨�����жϣ��������Ƿ����߶���(ʱ�䣬λ��, �Ƕ�)
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
Predict()���������������Ҫ�������˶�ѧģ�ͣ������ǵĻ��֣���̼ƵĻ��֣����򵥵�Ԥ����̬ƽ�ƾ����Լ���ת����
Ԥ�����˺��ּ����˼��ٶ�����ٶȵĹ۲⡣
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
����ִ����predict������Ȼ��ʹ����̼Ƶ����ݣ��õ���һ����̼Ƶ�У����odometry_correction_��
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

  //��IMU�л�ȡ ����
  const double last_yaw = transform::GetYaw(imu_tracker_->orientation());

  imu_tracker_->Advance(time);
  if (time_ > common::Time::min()) {
    const double delta_t = common::ToSeconds(time - time_);

// Constant velocity model.����һ�� �����˶�ģ�ͣ�
// λ��= x0 + v*t;
    const Eigen::Vector3d translation =
        pose_estimate_.translation() +
        delta_t *
            Eigen::Vector3d(velocity_estimate_.x(), velocity_estimate_.y(), 0.);

// Use the current IMU tracker roll and pitch for gravity alignment, and
// apply its change in yaw.
// ��ת = 
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
