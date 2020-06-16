from gym.envs.registration import register

register(
    id='RPiLEDEnv-v0',
    entry_point='envs.rpi_led_env:RPiLEDEnv',
)