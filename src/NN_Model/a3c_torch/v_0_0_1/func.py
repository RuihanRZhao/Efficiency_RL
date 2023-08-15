import torch

def self_env():
    import src.game.factory as factory
    return factory


def action_prob_normalize(action_prob):
    # Normalize action_prob matrix to probabilities
    action_p = action_prob - action_prob.min()
    return action_p / action_p.sum()

def sample_action(action_prob):

    # Normalize action_prob matrix to probabilities
    action_prob_normalized = action_prob_normalize(action_prob)

    # Generating random indices based on the normalized action probabilities
    num_samples = 3 * 12
    random_indices = torch.multinomial(action_prob_normalized.view(-1), num_samples, replacement=True)

    # Reshaping and scaling the random indices to [0, 1000] range
    random_indices_reshaped = random_indices.view(3, 12)
    random_values = random_indices_reshaped * (10 / (action_prob.shape[1] - 1))
    return random_values.int()