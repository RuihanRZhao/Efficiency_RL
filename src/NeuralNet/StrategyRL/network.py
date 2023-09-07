from net_components import *


class StrategyRL_Network(nn.Module):
    """
    A neural network module for strategy-based reinforcement learning.

    Args:
        input_H_size (int): The size of the horizontal input.
        input_V_size (int): The size of the vertical input.
        num_actions (int): The number of possible actions.
        num_action_choice (int): The number of action choices (default is 8).
        IP_hidden_size (int): The hidden size for information processing (default is 16).
        AG_hidden_size (int): The hidden size for action generation (default is 32).
        AP_hidden_size (int): The hidden size for action probability (default is 64).
        IP_num_layers (int): The number of layers in the information processing LSTM (default is 2).
        AG_num_layers (int): The number of layers in the action generation LSTM (default is 2).

    """
    def __init__(
            self,
            input_H_size: int = 0, input_V_size: int = 0,
            num_actions: int = 0, num_action_choice: int = 8,
            IP_hidden_size: int = 16,   AG_hidden_size: int = 32, AP_hidden_size: int = 64,
            IP_num_layers: int = 2,     AG_num_layers: int = 2

    ):
        super(StrategyRL_Network, self).__init__()
        IP_input_size = input_V_size

        self.information_processing = Information_Processing(
            input_size=IP_input_size,
            hidden_size=IP_hidden_size,
            num_layers=IP_num_layers
        )
        AG_input_size = input_V_size + IP_hidden_size
        AG_output_size: int = num_action_choice
        self.action_generation = Action_Generation(
            input_size=AG_input_size,
            hidden_size=AG_hidden_size,
            output_size=AG_output_size,
            num_layers=AG_num_layers,
            num_actions=num_actions
        )
        self.action_probability = Action_Probability(
            action_generation_output_size=AG_output_size,
            information_processing_output_size=AG_input_size,
            hidden_size=AP_hidden_size,  # Adjust the hidden size as needed
            num_actions=num_actions
        )
        self.action_output = Action_Output(
            num_actions=num_actions
        )

    def forward(
            self,
            input_matrix,
    ):
        """
        Forward pass of the StrategyRL module.

        Args:
            input_matrix (torch.Tensor): Input matrix.

        Returns:
            torch.Tensor: Action output.
        """
        IP_output = self.information_processing(input_matrix)
        AG_output = self.action_generation(IP_output, input_matrix)
        AP_output = self.action_probability(AG_output, IP_output)
        AO_output = self.action_output(AG_output, AP_output)

        return {
            "IP": IP_output,
            "AG": AG_output,
            "AP": AP_output,
            "AO": AO_output,
        }
