# Rear wheels stepping onto platforms

The platform task adds `ascent_rear_step` to encourage each rear wheel to lift
over a riser and land on its upper tread. The two rear wheels may move in
sequence; there is no requirement to lift them together. Both front wheels must
provide stable support on the upper surface before positive shaping is paid.

The term uses the existing height scanner and wheel contact sensor only during
training. Actor observations, history dimensions, actions, and network shapes
are unchanged, so existing platform checkpoints can be used for fine-tuning.
Ascending-platform curriculum heights range from 0.30 to 1.00 m across 40 levels;
level zero samples approximately 0.30–0.3175 m. Initial levels remain 0–5.
Descent and other terrain groups retain their original height ranges. The same
rear-step reward applies throughout ascent; no additional height gate is needed.

Use the user-selected `compress_v1` branch of the robot model repository:

```bash
git clone --branch compress_v1 --single-branch https://github.com/DDTRobot/ddt_ros2_control.git ddt_ros2_control
```

The D1 input is `ddt_ros2_control/urdfs/d1_description/urdf/robot.urdf`.
The training launch records the exact model repository commit and branch.

## Current settings

| Component | Value | Meaning |
| --- | --- | --- |
| Clearance margin | 0.06 m | Wheel bottom must rise above the target tread before crossing its lip. |
| Clearance bonus | 0.25 per wheel | Total reward for new lift progress on one tread; holding or repeating a lift adds nothing. |
| Clean landing bonus | 1.0 per wheel | Paid once after clearing the lip, advancing onto the tread, and establishing stable support without a detected wall contact. |
| Wall force threshold | 20 N | Deadband on horizontal force after subtracting 0.5 times upward force. |
| Wall penalty scale | 0.5 | Multiplies excess force normalized by 100 N, capped at 3 per wheel. |
| Approach distance | 0.30 m | Start increasing the approach penalty when the wheel's leading edge is within this distance of the estimated lip. |
| Approach penalty scale | 2.0 | Multiplies proximity, remaining lift fraction, and forward wheel translation speed (capped at 1 m/s). |

Progress and landing bonuses are divided by the control timestep before
RewardManager integration, so these totals do not change with control rate.
The combined term is added after ordinary reward clipping, preserving both
negative contact penalties and positive event bonuses. The existing 200 N rear
horizontal-force penalty remains active as well.

A target is inferred from an upward height change ahead of each rear wheel and
latched in world coordinates. A detected wall contact disqualifies that wheel's
current attempt from the clean landing bonus. After freeing the wheel from
contact, further lift can still earn progress if the front wheels support the
body. Progress only pays above the highest lift already reached on that attempt,
including height gained while touching the wall. Thus recovery does not pay
retroactively for wall-assisted climbing or repeatedly lifting to the same height.
The wall penalty and the 6 cm pre-crossing clearance requirement for clean
landings remain active. Once the wheel reaches the upper tread, a higher tread
may start a new attempt. Per-wheel state is cleared on environment reset.

With stable front support, `unsafe_approach` penalizes translating a rear wheel
toward the lip before it has enough clearance. The penalty rises continuously
with proximity and disappears as the wheel reaches the clearance height.
Stopping, retreating, and purely lateral motion incur no approach penalty.
After a cleared wheel has fully crossed the lip, settling onto the tread is
allowed. Dropping below clearance before crossing makes further forward motion
subject to the penalty again. This is a timestep-integrated penalty and is
included in the combined unclipped rear-step reward.

## Ascent difficulty progression

Ascending-platform environments now need all of the following for promotion:

- Pass the existing distance threshold (4 m from the environment origin).
- Each rear wheel has at least one clean landing during the episode.
- Neither rear wheel has triggered the wall-contact estimate during the episode.
- The episode did not terminate from a failure condition (timeout is allowed).

The curriculum reads these flags before RewardManager resets them. They persist
across individual tread targets and are cleared only for the environments being
reset. A completed traversal with a wall hit keeps its level. The existing
short-distance demotion rule and the other terrain groups retain their behavior.
Both-wheel evidence may accumulate sequentially; the wheels need not lift or
land together. This rule does not prove every physical riser was cleared, since
roughness and the contact estimator still limit the underlying measurements.

## Validation and limitations

Tensor-level trajectory tests cover sequential rear-wheel steps, repeated lifts,
dirty versus clean landings, free lift after a wall hit, ongoing contact and
passive climbing, support history, consecutive steps, partial resets, world
transforms, and timestep scaling. A separate test exercises the actual
environment reward-combination method. These checks do not establish learned
behavior or physical success rates.

Wall contact is inferred from wheel net forces and nearby terrain geometry; it
is not a direct wheel-to-riser contact-pair measurement. The existing scanner has
0.1 m spacing, and rocky side walls and tread roughness introduce uncertainty.
The reward cannot guarantee zero contact, or make a platform reachable beyond
the robot's joint, torque, and support limits.

Evaluate fine-tuned policies at fixed physical step heights with the same
commands and randomization as the baseline. Inspect rear-wheel trajectories,
actual wall contacts, complete traversal, and landing forces rather than only
total reward. Component traces are logged under:

- `Episode/Episode_Reward/ascent_rear_step/clearance_progress`
- `Episode/Episode_Reward/ascent_rear_step/clean_landing`
- `Episode/Episode_Reward/ascent_rear_step/wall_contact`
- `Episode/Episode_Reward/ascent_rear_step/recovery_progress`
- `Episode/Episode_Reward/ascent_rear_step/unsafe_approach`
- `Episode/Episode_Reward/ascent_rear_step/wall_free_clean_pair`

`recovery_progress` is the part of `clearance_progress` earned after an earlier
wall hit. It is logged separately to diagnose recovery, and is not added to the
reward a second time. When comparing runs, note that the earlier implementation
disabled all positive shaping after a hit; its clearance trace excluded recovery.

`wall_free_clean_pair` is a separate 0/1 episode diagnostic: both rear wheels
have clean landing records and there has been no wall-contact flag. A later
hit revokes it. Its episode average includes the other terrain groups, and it
does not include the distance or failure checks used for actual promotion.
It is not part of the reward.

The reward component traces are weighted episode sums divided by the configured maximum
episode duration, following RewardManager conventions; they are not directly
conditional success rates. Training startup was additionally checked with 4,096
environments for two NP3O updates on an RTX 5060 Ti (16 GB): the rear-step terms
were active, optimizer updates completed, and checkpoints were saved. This is a
startup check, not a measurement of learned platform traversal performance.
