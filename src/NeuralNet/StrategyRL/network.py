import torch

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

        self.information_processing = Information_Processing(
            input_size=input_V_size,
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
            information_processing_output_size=IP_hidden_size,
            hidden_size=AP_hidden_size,
            information_seq_size=input_H_size,
            num_actions=num_actions
        )
        self.action_output = Action_Output(
            num_actions=num_actions
        )

    def forward(
            self,
            input_matrix,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        IP_output = self.information_processing(input_matrix)
        AG_output = self.action_generation(IP_output, input_matrix)
        AP_output = self.action_probability(AG_output, IP_output)
        AO_output = self.action_output(AG_output, AP_output)

        print(
            f"size_IP: {IP_output.shape}"
            f"size_AG: {AG_output.shape}"
            f"size_AP: {AP_output.shape}"
            f"size_AO: {AO_output.shape}"
        )
        return {
            "IP": IP_output,
            "AG": AG_output,
            "AP": AP_output,
            "AO": AO_output,
        }


if __name__ == "__main__":
    input_H_size = 10
    input_V_size = 5
    num_actions = 3
    num_action_choice = 8
    IP_hidden_size = 32
    AG_hidden_size = 32
    AP_hidden_size = 32
    IP_num_layers = 2
    AG_num_layers = 2

    # Create an instance of the StrategyRL_Network
    model = StrategyRL_Network(
        input_H_size=input_H_size,
        input_V_size=input_V_size,
        num_actions=num_actions,
        num_action_choice=num_action_choice,
        IP_hidden_size=IP_hidden_size,
        AG_hidden_size=AG_hidden_size,
        AP_hidden_size=AP_hidden_size,
        IP_num_layers=IP_num_layers,
        AG_num_layers=AG_num_layers
    )

    # Generate a random input tensor
    input_tensor = torch.randn(1, input_H_size, input_V_size)

    # Perform a forward pass through the model
    outputs = model(input_tensor)
    IP_output = outputs["IP"]
    AG_output = outputs["AG"]
    AP_output = outputs["AP"]
    AO_output = outputs["AO"]

    print("IP:", IP_output.shape)
    print("AG:", AG_output.shape)
    print("AP:", AP_output.shape)
    print("AO:", AO_output)
