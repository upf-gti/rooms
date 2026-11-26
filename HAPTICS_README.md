Here are the definitions for the haptics_config
{
  "max_temperature": Max temperature command for the peltier, depends on the metalness parameter (degrees).
  "min_temperature": Min temperature command for the peltier, depends on the metalness parameter (degrees).
  "max_rumble": Max rumble command for the haptic motoros, depends on the rouhness parameter(Hz, max 255)
  "min_rumble": Min rumble command for the haptic motoros, depends on the rouhness parameter(Hz, max 255)
  "rumble_strength": Rumble strenght for the motors (max 255)
  "rumble_difference": Stagger different frecuencies for each rumble motor, so that they avoid the same config (Hz)
  "haptics_speed_cap" : Max hand speed support by the haptics (meters per second).
  "haptics_speed_min_multipliyer": Multiplier for the rumble frequency depending on the velocity. Represents very little speed.
  "haptics_speed_max_multipliyer": Multiplier for the rumble frequency depending on the velocity. Represents the max multiplicaion of the frequency, when the controller is moving at the speed cap.
  "haptic_device_port": Port for the haptics.
  "haptic_device_IP": IP for the haptic device.
}
