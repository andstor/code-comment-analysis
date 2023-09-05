g_ave_reps = 'g_ave_reps'
g_max_reps = 'g_max_reps'
o_ave_reps = 'o_ave_reps'
o_max_reps = 'o_max_reps'

# Path: constants.py

cwd = '/Users/andrestorhaug/Code/Projects/code-comment-clustering/'

data_files = [
    cwd + '.data/raw/generated_comments_0_10000_slack_256_code_embed.csv',
    cwd + '.data/raw/generated_comments_1_10000_slack_256_code_embed.csv',
    cwd + '.data/raw/generated_comments_2_10000_slack_256_code_embed.csv',
    cwd + '.data/raw/generated_comments_3_4000_slack_256_code_embed.csv'
]

embedding_column_names = [ 'g_ave_reps', 'g_max_reps', 'o_ave_reps', 'o_max_reps']