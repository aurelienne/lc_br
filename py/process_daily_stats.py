from generate_glm_stats import GLM_stats
import sys

config_file = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3]

glm = GLM_stats(config_file)
glm.process_daily_stats(input_file, output_file)

