# Following tutorial from :
# https://www.datahubbs.com/building-custom-gym-environments-for-rl/
# OpenAI Gym: https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/5_custom_gym_env.ipynb
# https://github.com/openai/gym/blob/master/docs/creating-environments.md
from setuptools import setup

setup(name='rpi_led_env',
      version='0.0.1',
      install_requires=['gym', 'opencv-python', 'numpy', 'matplotlib']
)